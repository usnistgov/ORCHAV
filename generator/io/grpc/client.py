#!/usr/bin/env python3
"""Diagnostic client for generator live-gRPC frames.

The production visualizer has its own frame-source code. This client is a small
generator-side helper for smoke tests and manual diagnostics. It requests
``FrameData`` messages from ``GeneratorService`` and decodes the embedded
``StandardMPCFrame`` protobuf into the shared compact frame contract.
"""

import logging
from typing import Any, Dict, List, Optional

import grpc

from shared.frames.protobuf import standard_mpc_frame_from_proto
from shared.frames.types import StandardMPCFrame
from shared.grpc_transport import GRPC_MESSAGE_OPTIONS
from shared.logging import get_logger

# Import the gRPC generated code
try:
    from shared.protos import visualizer_pb2 as _visualizer_pb2
    from shared.protos import visualizer_pb2_grpc as _visualizer_pb2_grpc

    visualizer_pb2: Any = _visualizer_pb2
    visualizer_pb2_grpc: Any = _visualizer_pb2_grpc
except ImportError as e:
    logging.error(f"Could not import gRPC generated code: {e}")
    logging.error("Make sure the protobuf files are generated.")
    raise

logger = get_logger(__name__)


class VisualizerGRPCClient:
    """Lightweight diagnostic client for the live generator gRPC service."""

    def __init__(self, server_address: str = "localhost:50051", timeout: float = 10.0):
        self.server_address = server_address
        self.timeout = timeout
        self.channel: Any = None
        self.stub: Any = None
        self.connected = False
        logger.info("Initializing Visualizer gRPC client for %s", server_address)

    def connect(self) -> bool:
        """Connect within the configured timeout.

        A failed setup returns ``False`` and clears any partially initialized
        channel state.
        """
        try:
            self.channel = grpc.insecure_channel(
                self.server_address,
                options=GRPC_MESSAGE_OPTIONS,
            )
            grpc.channel_ready_future(self.channel).result(timeout=self.timeout)
            self.stub = visualizer_pb2_grpc.GeneratorServiceStub(self.channel)
            self.connected = True
            logger.info("Connected to Generator gRPC server at %s", self.server_address)
            return True
        except grpc.FutureTimeoutError as exc:
            logger.error("Timed out connecting to %s: %s", self.server_address, exc)
            self.disconnect()
            return False
        except grpc.RpcError as exc:
            logger.error("gRPC connection failed: %s", exc)
            self.disconnect()
            return False
        except (OSError, RuntimeError) as exc:
            logger.error("Connection failed: %s", exc)
            self.disconnect()
            return False

    def disconnect(self) -> None:
        if self.channel:
            self.channel.close()
        self.channel = None
        self.stub = None
        self.connected = False
        logger.info("Disconnected from Generator gRPC server")

    def is_connected(self) -> bool:
        return self.connected and self.channel is not None

    def _invoke_stream(self, requests):
        if not self.stub:
            raise RuntimeError("Client not connected")
        call = self.stub.StreamFrames(iter(requests), timeout=self.timeout)
        try:
            for response in call:
                yield response
        finally:
            try:
                call.cancel()
            except (OSError, RuntimeError) as exc:
                logger.debug("Failed to cancel gRPC stream: %s", exc)

    def get_frame(self, frame_idx: int) -> Optional[StandardMPCFrame]:
        """Return one frame, or ``None`` when disconnected or rejected by the server."""
        if not self.is_connected():
            logger.warning("Cannot get frame - not connected to server")
            return None

        request = visualizer_pb2.FrameRequest(
            get_frame=visualizer_pb2.StreamFrameCommand(frame_idx=frame_idx)
        )

        for response in self._invoke_stream([request]):
            resp_type = response.WhichOneof("response_type")
            if resp_type == "frame_data":
                if int(response.frame_idx) != frame_idx:
                    raise ValueError(
                        f"Response frame {response.frame_idx} does not match "
                        f"requested frame {frame_idx}"
                    )
                return self._convert_from_protobuf_frame(
                    response.frame_data, expected_frame_idx=frame_idx
                )
            if resp_type == "error":
                logger.warning(
                    "Server reported error for frame %s: %s",
                    response.frame_idx,
                    response.error.message,
                )
                return None
        return None

    def get_frame_range(
        self, start_frame: int, end_frame: int, step: int = 1
    ) -> List[StandardMPCFrame]:
        """Return available frames from an end-exclusive index range.

        Results preserve request order. Frames rejected by the server are
        logged and omitted from the returned list.
        """
        if not self.is_connected():
            logger.warning("Cannot get frame range - not connected to server")
            return []

        frame_indices = list(range(start_frame, end_frame, step))
        requests = [
            visualizer_pb2.FrameRequest(get_frame=visualizer_pb2.StreamFrameCommand(frame_idx=idx))
            for idx in frame_indices
        ]

        frames: Dict[int, StandardMPCFrame] = {}
        for response in self._invoke_stream(requests):
            resp_type = response.WhichOneof("response_type")
            if resp_type == "frame_data":
                response_frame_idx = int(response.frame_idx)
                if response_frame_idx not in frame_indices:
                    raise ValueError(f"Response frame {response_frame_idx} was not requested")
                frames[response_frame_idx] = self._convert_from_protobuf_frame(
                    response.frame_data,
                    expected_frame_idx=response_frame_idx,
                )
                if len(frames) == len(frame_indices):
                    break
            elif resp_type == "error":
                logger.warning(
                    "Server reported error for frame %s: %s",
                    response.frame_idx,
                    response.error.message,
                )

        return [frames[idx] for idx in frame_indices if idx in frames]

    def get_frame_with_overrides(
        self,
        frame_idx: int,
        overrides: List[Dict[str, Any]],
    ) -> Optional[StandardMPCFrame]:
        """Send node overrides immediately before a frame request on one stream."""
        if not self.is_connected():
            logger.warning("Cannot get frame with overrides - not connected to server")
            return None

        type_mapping = {
            "tx": visualizer_pb2.NODE_TYPE_TX,
            "rx": visualizer_pb2.NODE_TYPE_RX,
            "target": visualizer_pb2.NODE_TYPE_TARGET,
        }
        override_messages = []
        for override in overrides:
            category = override.get("type", "").lower()
            node_type = type_mapping.get(category, visualizer_pb2.NODE_TYPE_UNSPECIFIED)
            position = override.get("position", [0.0, 0.0, 0.0])
            orientation = override.get("orientation", [])
            override_messages.append(
                visualizer_pb2.NodeOverride(
                    name=override.get("name", ""),
                    type=node_type,
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                    orientation=[float(o) for o in orientation],
                )
            )

        requests = [
            visualizer_pb2.FrameRequest(
                override_cmd=visualizer_pb2.NodeOverrideList(overrides=override_messages)
            ),
            visualizer_pb2.FrameRequest(
                get_frame=visualizer_pb2.StreamFrameCommand(frame_idx=frame_idx)
            ),
        ]

        for response in self._invoke_stream(requests):
            resp_type = response.WhichOneof("response_type")
            if resp_type == "frame_data":
                if int(response.frame_idx) != frame_idx:
                    raise ValueError(
                        f"Response frame {response.frame_idx} does not match "
                        f"requested frame {frame_idx}"
                    )
                return self._convert_from_protobuf_frame(
                    response.frame_data, expected_frame_idx=frame_idx
                )
            if resp_type == "error":
                logger.warning(
                    "Server reported error for frame %s with overrides: %s",
                    response.frame_idx,
                    response.error.message,
                )
                return None
        return None

    def get_frame_info(self) -> Optional[Dict[str, Any]]:
        if not self.is_connected():
            logger.warning("Cannot get frame info - not connected to server")
            return None
        try:
            request = visualizer_pb2.GetFrameInfoRequest()
            response = self.stub.GetFrameInfo(request, timeout=self.timeout)
            if not response.success:
                logger.warning("Server reported failure for frame info: %s", response.message)
                return None
            return {
                "total_frames": response.total_frames,
                "duration": response.duration,
                "frame_rate": response.frame_rate,
                "available_frames": list(response.available_frames),
            }
        except grpc.RpcError as exc:
            logger.error("gRPC error getting frame info: %s - %s", exc.code(), exc.details())
            return None
        except (OSError, RuntimeError) as exc:
            logger.error("Error getting frame info: %s", exc)
            return None

    def get_generator_status(self) -> Optional[Dict[str, Any]]:
        if not self.is_connected():
            logger.warning("Cannot get generator status - not connected to server")
            return None
        try:
            request = visualizer_pb2.GetGeneratorStatusRequest()
            response = self.stub.GetGeneratorStatus(request, timeout=self.timeout)
            if not response.success:
                logger.warning("Server reported failure for generator status: %s", response.message)
                return None
            return {
                "is_ready": response.is_ready,
                "is_streaming": response.is_streaming,
                "frames_generated": response.frames_generated,
                "uptime": response.uptime,
                "config": {
                    "data_mode": response.config.data_mode,
                    "motion_mode": response.config.motion_mode,
                    "num_steps": response.config.num_steps,
                    "duration": response.config.duration,
                    "output_mode": response.config.output_mode,
                    "enabled_patterns": list(response.config.enabled_patterns),
                },
            }
        except grpc.RpcError as exc:
            logger.error("gRPC error getting generator status: %s - %s", exc.code(), exc.details())
            return None
        except (OSError, RuntimeError) as exc:
            logger.error("Error getting generator status: %s", exc)
            return None

    def _convert_from_protobuf_frame(
        self,
        frame_pb: Any,
        *,
        expected_frame_idx: int,
    ) -> StandardMPCFrame:
        """Decode the required ``standard_mpc_frame`` protobuf payload."""
        if not frame_pb.HasField("standard_mpc_frame"):
            raise ValueError("FrameData is missing required standard_mpc_frame payload")
        frame = standard_mpc_frame_from_proto(frame_pb.standard_mpc_frame)
        if frame.frame_index != expected_frame_idx:
            raise ValueError(
                f"Encoded frame index {frame.frame_index} does not match "
                f"response frame {expected_frame_idx}"
            )
        return frame

    def __enter__(self):
        if not self.connect():
            raise ConnectionError(f"Could not connect to gRPC server at {self.server_address}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualizer gRPC Client")
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="Generator server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--test",
        choices=["connection", "frame", "range", "info", "status"],
        default="connection",
        help="Test to run (default: connection)",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index to test (default: 0)")
    parser.add_argument(
        "--start", type=int, default=0, help="Start frame for range test (default: 0)"
    )
    parser.add_argument("--end", type=int, default=5, help="End frame for range test (default: 5)")

    args = parser.parse_args()

    logger.info("Visualizer gRPC client test")
    logger.info("=" * 40)

    with VisualizerGRPCClient(args.server) as client:
        if args.test == "connection":
            logger.info("Connection test passed")
        elif args.test == "frame":
            frame_data = client.get_frame(args.frame)
            if frame_data:
                logger.info(
                    "Frame %d retrieved: %d TX, %d RX",
                    args.frame,
                    frame_data.num_tx,
                    frame_data.num_rx,
                )
            else:
                logger.error("Failed to retrieve frame %s", args.frame)
        elif args.test == "range":
            frames_data = client.get_frame_range(args.start, args.end)
            logger.info(
                "Retrieved %d frames from %d to %d",
                len(frames_data),
                args.start,
                args.end,
            )
        elif args.test == "info":
            info = client.get_frame_info()
            if info:
                logger.info(
                    "Frame info: %d frames, %.2f FPS",
                    info["total_frames"],
                    info["frame_rate"],
                )
            else:
                logger.error("Failed to retrieve frame info")
        elif args.test == "status":
            status = client.get_generator_status()
            if status:
                logger.info(
                    "Generator status: %d frames generated, uptime: %.2fs",
                    status["frames_generated"],
                    status["uptime"],
                )
            else:
                logger.error("Failed to retrieve generator status")
