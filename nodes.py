import os, sys
package_dir_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(package_dir_dir)

from .infer import infer

class XX_GVHMR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}) 
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "process"
    CATEGORY = "XX-GVHMR"

    def process(self, video_path=''):
        if video_path != '':
            out_file = os.path.join(
                os.path.dirname(video_path),
                os.path.splitext(os.path.basename(video_path))[0],
                '1_incam2.mp4'
                )
            if not os.path.exists(out_file): 
                infer(video_path=video_path)
            return (out_file,)

NODE_CLASS_MAPPINGS = {    
    "XX_GVHMR":XX_GVHMR,
}

NODE_DISPLAY_NAME_MAPPINGS = {    
    "XX GVHMR":"XX_GVHMR",
}