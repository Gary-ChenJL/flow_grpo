"""
VideoAlign Reward Scorer
Based on VideoAlign/VideoReward from https://github.com/KlingTeam/VideoAlign
Uses Qwen2-VL-2B-Instruct based reward model for video quality assessment
"""

import torch
import numpy as np
import tempfile
import os
from pathlib import Path

class VideoAlignScorer(torch.nn.Module):
    def __init__(self, checkpoint_path="hf_cache/VideoReward", device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Lazy initialization of the VideoAlign model"""
        try:
            # Import VideoAlign inference class
            # This assumes VideoAlign is installed or available in the path
            import sys
            videoalign_path = Path(__file__).parent.parent / "VideoAlign"
            if videoalign_path.exists():
                sys.path.insert(0, str(videoalign_path))

            from inference import VideoVLMRewardInference

            self.model = VideoVLMRewardInference(
                load_from_pretrained=self.checkpoint_path,
                device=self.device,
                dtype=self.dtype
            )
            print(f"VideoAlign model loaded from {self.checkpoint_path}")
        except ImportError as e:
            print(f"Warning: Could not import VideoAlign. Error: {e}")
            print("Please ensure VideoAlign is installed: git clone https://github.com/KlingTeam/VideoAlign")
            self.model = None
        except Exception as e:
            print(f"Warning: Could not load VideoAlign model. Error: {e}")
            self.model = None

    def _save_video_tensor_to_file(self, video_tensor, filepath, fps=8):
        """
        Save a video tensor to an MP4 file
        Args:
            video_tensor: torch.Tensor of shape (F, H, W, C) or (F, C, H, W)
            filepath: str, output path
            fps: int, frames per second
        """
        try:
            import cv2
        except ImportError:
            print("Warning: opencv-python not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'opencv-python'])
            import cv2

        # Convert tensor to numpy and ensure correct format
        if isinstance(video_tensor, torch.Tensor):
            video_np = video_tensor.cpu().numpy()
        else:
            video_np = video_tensor

        # Handle different tensor formats
        if video_np.shape[1] == 3 or video_np.shape[1] == 1:  # (F, C, H, W)
            video_np = video_np.transpose(0, 2, 3, 1)  # -> (F, H, W, C)

        # Ensure uint8 format
        if video_np.dtype != np.uint8:
            video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

        # Get dimensions
        num_frames, height, width, channels = video_np.shape

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        # Write frames
        for frame in video_np:
            if channels == 3:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)

        out.release()

    @torch.no_grad()
    def __call__(self, videos, prompts, return_details=False):
        """
        Score videos using VideoAlign reward model

        Args:
            videos: torch.Tensor of shape (B, F, C, H, W) or (B, F, H, W, C)
            prompts: list of str, text prompts for each video
            return_details: bool, if True return dict with VQ, MQ, TA scores

        Returns:
            scores: list of float (Overall scores) or list of dict if return_details=True
        """
        if self.model is None:
            # Fallback: return dummy scores if model not available
            print("Warning: VideoAlign model not available, returning dummy scores")
            if return_details:
                return [{"VQ": 0.5, "MQ": 0.5, "TA": 0.5, "Overall": 1.5} for _ in prompts]
            else:
                return [0.5] * len(prompts)

        # Convert videos to temporary files
        temp_files = []
        try:
            for i, video in enumerate(videos):
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    temp_path = tmp.name

                # Handle tensor format: (B, F, C, H, W) -> (F, C, H, W)
                if isinstance(video, torch.Tensor):
                    if video.dim() == 5:
                        video = video[0]  # Already indexed by batch

                    # Convert (F, C, H, W) to (F, H, W, C) if needed
                    if video.shape[1] == 3:  # (F, C, H, W)
                        video = video.permute(0, 2, 3, 1)  # -> (F, H, W, C)

                # Save video to temp file
                self._save_video_tensor_to_file(video, temp_path, fps=8)
                temp_files.append(temp_path)

            # Call VideoAlign model
            results = self.model.reward(temp_files, prompts, use_norm=True)

            # Extract scores
            if return_details:
                scores = results
            else:
                scores = [r['Overall'] for r in results]

            return scores

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_file}: {e}")


# Usage example
def main():
    scorer = VideoAlignScorer(
        checkpoint_path="hf_cache/VideoReward",
        device="cuda",
        dtype=torch.bfloat16
    )

    # Create dummy video tensor (B, F, C, H, W)
    batch_size = 2
    num_frames = 16
    height, width = 240, 416
    videos = torch.rand(batch_size, num_frames, 3, height, width)

    prompts = [
        "A cat playing with a ball",
        "A person walking in the park"
    ]

    # Get overall scores
    scores = scorer(videos, prompts)
    print(f"Overall scores: {scores}")

    # Get detailed scores
    detailed_scores = scorer(videos, prompts, return_details=True)
    print(f"Detailed scores: {detailed_scores}")


if __name__ == "__main__":
    main()
