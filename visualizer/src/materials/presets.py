"""Built-in visual PBR presets shared by profiles, UI, and resolution."""

BUILTIN_MATERIAL_PRESETS = {
    # === Surface Finish Presets ===
    "Mirror/Polished": {
        "description": "Highly polished mirror-like surface",
        "roughness": 0.02,
        "metallic": 0.0,
        "reflectance": 0.98,
        "alpha": 1.0,
    },
    "Glossy": {
        "description": "Smooth glossy finish like lacquered wood",
        "roughness": 0.15,
        "metallic": 0.0,
        "reflectance": 0.7,
        "alpha": 1.0,
    },
    "Satin": {
        "description": "Soft satin-like sheen",
        "roughness": 0.35,
        "metallic": 0.0,
        "reflectance": 0.5,
        "alpha": 1.0,
    },
    "Matte": {
        "description": "Flat matte finish, no shine",
        "roughness": 0.9,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
    },
    "Rough": {
        "description": "Very rough textured surface",
        "roughness": 1.0,
        "metallic": 0.0,
        "reflectance": 0.1,
        "alpha": 1.0,
    },
    # === Metal Presets ===
    "Polished Metal": {
        "description": "Shiny polished metal surface",
        "roughness": 0.08,
        "metallic": 1.0,
        "reflectance": 0.95,
        "alpha": 1.0,
    },
    "Brushed Metal": {
        "description": "Brushed metal with subtle texture",
        "roughness": 0.45,
        "metallic": 1.0,
        "reflectance": 0.7,
        "alpha": 1.0,
    },
    "Oxidized Metal": {
        "description": "Weathered oxidized metal",
        "roughness": 0.75,
        "metallic": 0.6,
        "reflectance": 0.3,
        "alpha": 1.0,
    },
    # === Glass/Transparent Presets ===
    "Clear Glass": {
        "description": "Transparent clear glass",
        "roughness": 0.02,
        "metallic": 0.0,
        "reflectance": 0.98,
        "alpha": 0.12,
    },
    "Frosted Glass": {
        "description": "Translucent frosted glass",
        "roughness": 0.5,
        "metallic": 0.0,
        "reflectance": 0.6,
        "alpha": 0.35,
    },
    "Tinted Glass": {
        "description": "Semi-transparent tinted glass",
        "roughness": 0.05,
        "metallic": 0.0,
        "reflectance": 0.8,
        "alpha": 0.25,
    },
    # === Special Effect Presets ===
    "Ghost/Wireframe": {
        "description": "Very transparent ghostly appearance",
        "roughness": 0.3,
        "metallic": 0.0,
        "reflectance": 0.4,
        "alpha": 0.08,
    },
    # === Colored Metal Presets ===
    "Gold": {
        "description": "Golden metallic finish",
        "roughness": 0.15,
        "metallic": 1.0,
        "reflectance": 0.9,
        "alpha": 1.0,
        "color": [1.0, 0.84, 0.0],  # Gold color
    },
    "Copper": {
        "description": "Warm copper metallic finish",
        "roughness": 0.2,
        "metallic": 1.0,
        "reflectance": 0.85,
        "alpha": 1.0,
        "color": [0.72, 0.45, 0.2],  # Copper color
    },
    "Bronze": {
        "description": "Rich bronze metallic finish",
        "roughness": 0.25,
        "metallic": 0.9,
        "reflectance": 0.75,
        "alpha": 1.0,
        "color": [0.55, 0.47, 0.33],  # Bronze color
    },
    "Silver": {
        "description": "Bright silver metallic finish",
        "roughness": 0.1,
        "metallic": 1.0,
        "reflectance": 0.95,
        "alpha": 1.0,
        "color": [0.85, 0.87, 0.9],  # Silver color
    },
    # === Colored Non-Metal Presets ===
    "Red Plastic": {
        "description": "Glossy red plastic",
        "roughness": 0.25,
        "metallic": 0.0,
        "reflectance": 0.5,
        "alpha": 1.0,
        "color": [0.9, 0.15, 0.1],  # Red
    },
    "Blue Glass": {
        "description": "Tinted blue glass",
        "roughness": 0.05,
        "metallic": 0.0,
        "reflectance": 0.85,
        "alpha": 0.2,
        "color": [0.2, 0.4, 0.9],  # Blue
    },
    "Green Matte": {
        "description": "Matte green surface",
        "roughness": 0.85,
        "metallic": 0.0,
        "reflectance": 0.25,
        "alpha": 1.0,
        "color": [0.2, 0.6, 0.25],  # Green
    },
    # === Building Material Presets ===
    "Concrete": {
        "description": "Standard concrete surface",
        "roughness": 0.85,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
    },
    "Asphalt": {
        "description": "Dark asphalt road surface",
        "roughness": 0.9,
        "metallic": 0.0,
        "reflectance": 0.15,
        "alpha": 1.0,
        "color": [0.15, 0.15, 0.15],  # Dark gray
    },
    "Brick": {
        "description": "Red brick material",
        "roughness": 0.8,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
        "color": [0.6, 0.25, 0.15],  # Brick red
    },
    "Stucco": {
        "description": "Textured exterior wall finish",
        "roughness": 0.95,
        "metallic": 0.0,
        "reflectance": 0.15,
        "alpha": 1.0,
        "color": [0.9, 0.85, 0.75],  # Warm beige
    },
    "Grass": {
        "description": "Natural grass ground surface",
        "roughness": 0.95,
        "metallic": 0.0,
        "reflectance": 0.1,
        "alpha": 1.0,
        "color": [0.2, 0.5, 0.15],  # Green
    },
    "Cobblestone": {
        "description": "Gray cobblestone pavement",
        "roughness": 0.8,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
        "color": [0.55, 0.52, 0.48],  # Warm gray
    },
    "NIST CTL Floor": {
        "description": "Presentation floor with embedded NIST/CTL sensing branding",
        "roughness": 0.72,
        "metallic": 0.02,
        "reflectance": 0.32,
        "alpha": 1.0,
        "clearcoat": 0.12,
        "clearcoat_roughness": 0.35,
        "anisotropy": 0.0,
        "color": [0.20, 0.20, 0.19],
    },
    "Plywood": {
        "description": "Light wood panel",
        "roughness": 0.75,
        "metallic": 0.0,
        "reflectance": 0.25,
        "alpha": 1.0,
        "color": [0.85, 0.7, 0.5],  # Light tan
    },
    "Ceramic Tile": {
        "description": "Glossy floor or wall tile",
        "roughness": 0.15,
        "metallic": 0.0,
        "reflectance": 0.7,
        "alpha": 1.0,
        "color": [0.9, 0.9, 0.88],  # Off-white
    },
    "Linoleum": {
        "description": "Vinyl floor covering",
        "roughness": 0.4,
        "metallic": 0.0,
        "reflectance": 0.3,
        "alpha": 1.0,
        "color": [0.6, 0.55, 0.5],  # Gray-brown
    },
    "Carpet": {
        "description": "Soft textile floor covering",
        "roughness": 1.0,
        "metallic": 0.0,
        "reflectance": 0.1,
        "alpha": 1.0,
        "color": [0.4, 0.35, 0.3],  # Dark brown
    },
    # === Stone Presets ===
    "Granite": {
        "description": "Speckled hard stone",
        "roughness": 0.4,
        "metallic": 0.0,
        "reflectance": 0.5,
        "alpha": 1.0,
        "color": [0.45, 0.45, 0.5],  # Gray with blue tint
    },
    "Limestone": {
        "description": "Light tan sedimentary stone",
        "roughness": 0.7,
        "metallic": 0.0,
        "reflectance": 0.3,
        "alpha": 1.0,
        "color": [0.9, 0.85, 0.7],  # Warm tan
    },
    "Sandstone": {
        "description": "Warm porous stone",
        "roughness": 0.85,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
        "color": [0.85, 0.7, 0.5],  # Sandy tan
    },
    "Slate": {
        "description": "Dark layered stone",
        "roughness": 0.6,
        "metallic": 0.0,
        "reflectance": 0.35,
        "alpha": 1.0,
        "color": [0.3, 0.32, 0.35],  # Dark blue-gray
    },
    "Terracotta": {
        "description": "Reddish clay material",
        "roughness": 0.8,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
        "color": [0.8, 0.45, 0.3],  # Orange-red
    },
    # === Additional Metal Presets ===
    "Aluminum": {
        "description": "Light brushed aluminum",
        "roughness": 0.35,
        "metallic": 1.0,
        "reflectance": 0.9,
        "alpha": 1.0,
        "color": [0.91, 0.92, 0.92],  # Light silver
    },
    "Steel": {
        "description": "Industrial steel",
        "roughness": 0.4,
        "metallic": 1.0,
        "reflectance": 0.75,
        "alpha": 1.0,
        "color": [0.55, 0.57, 0.58],  # Dark gray
    },
    "Chrome": {
        "description": "Mirror-like chrome plating",
        "roughness": 0.02,
        "metallic": 1.0,
        "reflectance": 0.98,
        "alpha": 1.0,
        "color": [0.95, 0.95, 0.97],  # Bright silver
    },
    "Rusty Metal": {
        "description": "Corroded weathered metal",
        "roughness": 0.9,
        "metallic": 0.4,
        "reflectance": 0.2,
        "alpha": 1.0,
        "color": [0.5, 0.3, 0.2],  # Brown-orange
    },
    "Titanium": {
        "description": "Strong lightweight metal",
        "roughness": 0.3,
        "metallic": 1.0,
        "reflectance": 0.8,
        "alpha": 1.0,
        "color": [0.6, 0.6, 0.65],  # Gray with slight blue
    },
    # === Fabric & Soft Materials ===
    "Rubber": {
        "description": "Black rubber material",
        "roughness": 0.9,
        "metallic": 0.0,
        "reflectance": 0.1,
        "alpha": 1.0,
        "color": [0.1, 0.1, 0.1],  # Near black
    },
    "Plastic": {
        "description": "Generic smooth plastic",
        "roughness": 0.35,
        "metallic": 0.0,
        "reflectance": 0.4,
        "alpha": 1.0,
        "color": [0.7, 0.7, 0.7],  # Light gray
    },
    "Fabric": {
        "description": "Woven textile material",
        "roughness": 0.95,
        "metallic": 0.0,
        "reflectance": 0.1,
        "alpha": 1.0,
        "color": [0.5, 0.45, 0.4],  # Neutral brown
    },
    "Leather": {
        "description": "Smooth leather surface",
        "roughness": 0.6,
        "metallic": 0.0,
        "reflectance": 0.25,
        "alpha": 1.0,
        "color": [0.35, 0.2, 0.1],  # Dark brown
    },
    "Skin": {
        "description": "Human skin with subtle subsurface sheen",
        "roughness": 0.55,
        "metallic": 0.0,
        "reflectance": 0.35,
        "alpha": 1.0,
        "color": [0.82, 0.62, 0.49],  # Warm peach
    },
    "Paint": {
        "description": "Painted wall surface",
        "roughness": 0.5,
        "metallic": 0.0,
        "reflectance": 0.3,
        "alpha": 1.0,
        "color": [0.95, 0.95, 0.93],  # Off-white
    },
    # === Environmental/Weather Presets ===
    "Snow": {
        "description": "Fresh white snow",
        "roughness": 0.8,
        "metallic": 0.0,
        "reflectance": 0.9,
        "alpha": 1.0,
        "color": [0.98, 0.98, 1.0],  # Pure white
    },
    "Ice": {
        "description": "Frozen ice surface",
        "roughness": 0.1,
        "metallic": 0.0,
        "reflectance": 0.8,
        "alpha": 0.8,
        "color": [0.85, 0.92, 0.98],  # Light blue
    },
    "Sand": {
        "description": "Beach or desert sand",
        "roughness": 1.0,
        "metallic": 0.0,
        "reflectance": 0.15,
        "alpha": 1.0,
        "color": [0.9, 0.8, 0.6],  # Warm tan
    },
}
