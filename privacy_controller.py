PRIVACY_MODES = {

    "natural": {

        "name": "Natural Appearance",

        # Geometry
        "warp_strength": 0.35,

        # Texture
        "texture_strength": 0.10,

        # Elastic
        "elastic_strength": 0.00,

        # Identity texture
        "identity_texture_strength": 0.10,

        # Blur
        "blur_strength": 0,

        # Modules
        "enable_texture": False,
        "enable_elastic": False,
        "enable_identity_texture": True,

    },

    "balanced": {

        "name": "Balanced Protection",

        "warp_strength": 0.70,

        "texture_strength": 0.20,

        "elastic_strength": 0.15,

        "identity_texture_strength": 0.20,

        "blur_strength": 0,

        "enable_texture": True,
        "enable_elastic": True,
        "enable_identity_texture": True,

    },

    "strong": {

        "name": "Enhanced Privacy",

        "warp_strength": 1.00,

        "texture_strength": 0.35,

        "elastic_strength": 0.30,

        "identity_texture_strength": 0.35,

        "blur_strength": 0,

        "enable_texture": True,
        "enable_elastic": True,
        "enable_identity_texture": True,

    },

    "maximum": {

        "name": "Maximum Privacy",

        # Don't over-warp geometry
        "warp_strength": 1.15,

        # Most privacy comes from texture
        "texture_strength": 0.55,

        "elastic_strength": 0.45,

        "identity_texture_strength": 0.50,

        "blur_strength": 1,

        "enable_texture": True,
        "enable_elastic": True,
        "enable_identity_texture": True,

    }

}


def get_privacy_settings(mode):

    return PRIVACY_MODES.get(
        mode,
        PRIVACY_MODES["balanced"]
    )