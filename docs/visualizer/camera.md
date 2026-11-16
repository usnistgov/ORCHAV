# Camera navigation and recording

The pygfx camera can follow actors, show their point of view, orbit a scene and
replay saved viewpoints. These controls change the display without changing
actor poses, propagation paths or recorded timing. The
[Eiffel Tower demo](demo_video.md#eiffel-tower-scene-and-propagation-paths) shows
a camera tour followed by a fixed view of recorded motion.

The main **Camera** panel selects the mode and actor. Enable **Advanced camera**
to expose **Navigate**, **Framing**, and **Views & sequence** under **Rendering**.
Use **Stop camera motion** or **Escape** to stop an active animation.

## Navigate while replaying

| Control | Behavior |
|---|---|
| **Follow** | Moves with the selected actor while allowing orbit, zoom and pan. A pan offsets the framing. |
| **POV** | Places the camera on the actor and uses its selected look direction. Mouse input or Escape returns to Overview. |
| **Smooth transitions** | Eases changes between actors, modes and views over a duration in seconds. Mouse input or Escape interrupts the transition. |
| **Previous view** | Returns to the pose before a deliberate navigation action. |
| **Focus** | Frames the selected actor, transmitter–receiver pair or inspected coverage cell. Select a specific TX and RX before focusing their pair. |

Focus can also frame a saved sensing position or track. Select it in the sensing
inspector first. A range-only result does not specify a 3D position, and the
panel explains when focus is unavailable.

## Rotate around a scene or actor

In **Navigate**, select **Scene center**, **Selected actor**, **Current look-at**
or a **World point** in metres, then choose **Start orbit**. Speed is in degrees
per second. Positive values rotate counterclockwise when viewed from above.
An actor orbit moves with that actor during playback.

Orbiting uses wall time and also works while data playback is paused. It rotates
about world Z while preserving height and radius. Drag, scroll or press Escape
to take control. Opening another recording stops the motion.

## Follow framing and POV lens

**Framing → Follow behavior** offers three choices:

| Choice | Framing |
|---|---|
| **Keep world heading** | Follows position without rotating the camera offset with the actor. |
| **Behind actor (chase)** | Uses the actor's recorded yaw to stay behind it. Requires recorded orientation. |
| **Fixed position, track actor** | Keeps the camera position and changes only its viewing direction. |

**Frame behind selected actor** enables chase. Set **Distance behind**,
**Side offset (left)** and **Height above actor** in metres. Positive values
mean behind, left and above. Negative values reverse those directions. Chase
measures offsets relative to actor yaw. World-heading Follow uses a fixed
horizontal heading from the camera view. Mouse zoom and orbit remain available.

To watch an actor pass a fixed camera, place the view and choose **Track actor
from here**. The current camera position and field of view are retained, and
offset controls are disabled. Drag the view to relocate that fixed camera.

POV offsets use the actor's local forward, left and up directions in metres.
**True POV** starts enabled, placing the camera at the actor's reference position
and hiding that actor to avoid viewing its interior. Disable it to apply offsets
and restore normal actor visibility. Re-enabling it preserves the offsets for
later reuse.

Set the field of view in degrees. **Keep POV horizon level** uses world Z-up.
Disable it to let actor roll affect the view. Framing changes apply immediately,
including while data playback is paused.

## Save and replay camera views

Under **Views & sequence**, name the view and choose **Save / replace named
view**. A saved view is world-fixed unless **Translate with selected actor** is
checked. Actor-relative views translate with that actor but do not rotate with
its heading.

Views, framing settings and sequences are retained in a
[workspace snapshot](workspace_snapshots.md). **Import / Export camera JSON**
shares camera settings alone. It does not carry scene data, and referenced
actors must exist in the receiving recording. Opening another recording clears
the current views until its own workspace or camera settings are loaded.

To create a sequence:

1. Select a named view, set **Hold view** and **Transition into view** in seconds,
   and append it.
2. Add more views. Use **Up**, **Down** or **Remove** to edit their order. The
   first view appears immediately, so its transition duration is ignored.
3. Choose **Play sequence**. **Loop camera sequence** repeats it. The main
   playback controls independently determine whether actors move.

Transitions interpolate the camera position, orientation and lens. They do not
avoid buildings automatically, so choose intermediate views for an unobstructed
camera path. The four quick numbered slots provide temporary viewpoints.

## Export a camera sequence

Choose **Export camera sequence as MP4 / GIF…**, or enable **Use saved camera
sequence** in the video export dialog.

| Setting | Meaning |
|---|---|
| **Hold current data frame** | Default. Only the camera moves around the current scene state. |
| Selected frame range | With Hold disabled, distribute those recorded frames over the camera sequence. Frames are stepped or repeated, never interpolated. |
| FPS | Camera samples per second. The export contains `ceil(duration × FPS)` images. |

Camera timing is presentation timing, independent of the simulation sampling
rate. Export pauses camera and data playback and restores their prior state
when it finishes or is cancelled.

Install the optional MP4 encoder if needed:

```bash
python -m pip install 'imageio[ffmpeg]'
```

GIF uses ImageIO and Pillow. Start with a modest resolution and duration.
Exported videos and camera JSON do not contain the underlying simulation data.
Use [Demo mode](presentation_tools.md#demo-mode) to combine camera motion with
automatic appearance changes or saved sensing explanations.

Up: [Visualizer](README.md) | Related: [Workspace snapshots](workspace_snapshots.md) |
[Presentation tools](presentation_tools.md)
