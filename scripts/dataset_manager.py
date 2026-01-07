"""Dataset Manager for LTX-Video-Trainer.

This module provides utilities for managing video datasets, including:
- Creating and organizing datasets
- Adding videos with automatic thumbnail generation
- Managing captions
- Dataset validation
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2


class DatasetManager:
    """Manager for video dataset operations."""

    def __init__(self, datasets_root: Path = Path(__file__).parent.parent / "datasets"):
        """Initialize the dataset manager.

        Args:
            datasets_root: Root directory for all datasets
        """
        self.datasets_root = datasets_root
        self.datasets_root.mkdir(exist_ok=True)

    def list_datasets(self) -> list[str]:
        """Get all dataset names.

        Returns:
            List of dataset names
        """
        return [d.name for d in self.datasets_root.iterdir() if d.is_dir()]

    def create_dataset(self, name: str) -> Path:
        """Create new dataset directory structure.

        Args:
            name: Name of the dataset

        Returns:
            Path to the created dataset directory

        Raises:
            ValueError: If dataset name is invalid or already exists
        """
        if not name or not name.strip():
            raise ValueError("Dataset name cannot be empty")

        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
        if not safe_name:
            raise ValueError("Dataset name must contain alphanumeric characters")

        dataset_dir = self.datasets_root / safe_name

        if dataset_dir.exists():
            raise ValueError(f"Dataset '{safe_name}' already exists")

        (dataset_dir / "videos").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "thumbnails").mkdir(parents=True, exist_ok=True)

        dataset_json = dataset_dir / "dataset.json"
        with open(dataset_json, "w") as f:
            json.dump([], f)

        return dataset_dir

    def add_videos(
        self, dataset_name: str, video_files: list[str], reference_files: Optional[list[str]] = None
    ) -> dict:
        """Add videos to dataset and generate thumbnails.

        Args:
            dataset_name: Name of the dataset
            video_files: List of video file paths to add
            reference_files: Optional list of reference video files for IC-LoRA (must match video_files length)

        Returns:
            Dictionary with 'added' and 'failed' lists
        """
        dataset_dir = self.datasets_root / dataset_name
        if not dataset_dir.exists():
            raise ValueError(f"Dataset '{dataset_name}' does not exist")

        videos_dir = dataset_dir / "videos"
        thumbs_dir = dataset_dir / "thumbnails"

        if reference_files:
            references_dir = dataset_dir / "references"
            references_dir.mkdir(exist_ok=True)

            if len(reference_files) != len(video_files):
                raise ValueError("Number of reference videos must match number of videos")

        results = {"added": [], "failed": []}

        dataset_json = dataset_dir / "dataset.json"
        with open(dataset_json) as f:
            items = json.load(f)

        for idx, video_file in enumerate(video_files):
            try:
                video_path = Path(video_file)
                dest_path = videos_dir / video_path.name

                if dest_path.exists():
                    results["failed"].append({"video": video_path.name, "error": "Already exists"})
                    continue

                shutil.copy2(video_file, dest_path)

                thumb_path = self._generate_thumbnail(dest_path, thumbs_dir)

                entry = {"media_path": f"videos/{video_path.name}", "caption": ""}

                if reference_files and idx < len(reference_files):
                    ref_path = Path(reference_files[idx])
                    ref_dest_path = references_dir / ref_path.name
                    shutil.copy2(reference_files[idx], ref_dest_path)
                    entry["reference_path"] = f"references/{ref_path.name}"

                items.append(entry)

                results["added"].append({"video": video_path.name, "thumbnail": thumb_path.name})

            except Exception as e:
                results["failed"].append({"video": Path(video_file).name, "error": str(e)})

        with open(dataset_json, "w") as f:
            json.dump(items, f, indent=2)

        return results

    def _generate_thumbnail(self, video_path: Path, output_dir: Path) -> Path:
        """Extract first frame as thumbnail.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save thumbnail

        Returns:
            Path to the generated thumbnail

        Raises:
            ValueError: If video cannot be read
        """
        cap = cv2.VideoCapture(str(video_path))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read video: {video_path}")

        thumb_path = output_dir / f"{video_path.stem}.jpg"

        height, width = frame.shape[:2]
        target_size = 256

        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))

        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if new_width != new_height:
            max_dim = max(new_width, new_height)
            square_frame = cv2.copyMakeBorder(
                frame,
                (max_dim - new_height) // 2,
                (max_dim - new_height + 1) // 2,
                (max_dim - new_width) // 2,
                (max_dim - new_width + 1) // 2,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            frame = square_frame

        cv2.imwrite(str(thumb_path), frame)
        return thumb_path

    def get_dataset_items(self, dataset_name: str) -> list[dict]:
        """Get all videos and captions in dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of dictionaries with 'media_path', 'caption', and 'thumbnail'
        """
        dataset_dir = self.datasets_root / dataset_name
        if not dataset_dir.exists():
            return []

        dataset_json = dataset_dir / "dataset.json"

        if dataset_json.exists():
            with open(dataset_json) as f:
                items = json.load(f)
        else:
            items = []

        for item in items:
            video_name = Path(item["media_path"]).name
            thumb_path = dataset_dir / "thumbnails" / f"{Path(video_name).stem}.jpg"
            item["thumbnail"] = str(thumb_path) if thumb_path.exists() else None
            item["full_video_path"] = str(dataset_dir / item["media_path"])

        return items

    def update_caption(self, dataset_name: str, video_name: str, caption: str) -> None:
        """Update caption for a specific video.

        Args:
            dataset_name: Name of the dataset
            video_name: Name of the video file
            caption: New caption text
        """
        dataset_dir = self.datasets_root / dataset_name
        dataset_json = dataset_dir / "dataset.json"

        with open(dataset_json) as f:
            items = json.load(f)

        found = False
        for item in items:
            if Path(item["media_path"]).name == video_name:
                item["caption"] = caption
                found = True
                break

        if not found:
            items.append({"media_path": f"videos/{video_name}", "caption": caption})

        with open(dataset_json, "w") as f:
            json.dump(items, f, indent=2)

    def delete_video(self, dataset_name: str, video_name: str) -> None:
        """Remove video from dataset.

        Args:
            dataset_name: Name of the dataset
            video_name: Name of the video file to delete
        """
        dataset_dir = self.datasets_root / dataset_name

        dataset_json = dataset_dir / "dataset.json"
        with open(dataset_json) as f:
            items = json.load(f)

        items = [i for i in items if Path(i["media_path"]).name != video_name]

        with open(dataset_json, "w") as f:
            json.dump(items, f, indent=2)

        (dataset_dir / "videos" / video_name).unlink(missing_ok=True)
        (dataset_dir / "thumbnails" / f"{Path(video_name).stem}.jpg").unlink(missing_ok=True)

    def get_dataset_stats(self, dataset_name: str) -> dict:
        """Get statistics about dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset statistics
        """
        dataset_dir = self.datasets_root / dataset_name
        items = self.get_dataset_items(dataset_name)

        has_references = any(i.get("reference_path") for i in items)
        precomputed_dir = dataset_dir / ".precomputed"
        has_reference_latents = (precomputed_dir / "reference_latents").exists() if precomputed_dir.exists() else False

        return {
            "name": dataset_name,
            "total_videos": len(items),
            "captioned": sum(1 for i in items if i.get("caption") and i.get("caption").strip()),
            "uncaptioned": sum(1 for i in items if not i.get("caption") or not i.get("caption").strip()),
            "preprocessed": precomputed_dir.exists(),
            "has_references": has_references,
            "reference_latents_computed": has_reference_latents,
        }

    def validate_dataset(self, dataset_name: str) -> dict:
        """Validate dataset and return issues.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        items = self.get_dataset_items(dataset_name)
        dataset_dir = self.datasets_root / dataset_name

        if not items:
            issues.append("Dataset is empty")
            return {"valid": False, "issues": issues, "warnings": warnings, "total_videos": 0}

        uncaptioned = [i for i in items if not i.get("caption") or not i.get("caption").strip()]
        if uncaptioned:
            warnings.append(f"{len(uncaptioned)} videos without captions")

        for item in items:
            video_path = dataset_dir / item["media_path"]
            if not video_path.exists():
                issues.append(f"Missing video: {item['media_path']}")

        if not (dataset_dir / ".precomputed").exists():
            warnings.append("Dataset not preprocessed - will be slower to train")

        for item in items[:5]:
            video_path = dataset_dir / item["media_path"]
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    issues.append(f"Cannot read video: {video_path.name}")
                cap.release()

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_videos": len(items),
        }

    def split_video_scenes(
        self,
        video_path: Path,
        output_dir: Path,
        min_scene_length: Optional[int] = None,
        threshold: Optional[float] = None,
        detector_type: str = "content",
        max_scenes: Optional[int] = None,
        filter_shorter_than: Optional[str] = None,
    ) -> list[Path]:
        """Split a video into scenes using advanced scene detection.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save split scenes
            min_scene_length: Minimum scene length in frames
            threshold: Detection threshold
            detector_type: Type of detector ('content', 'adaptive', 'threshold', 'histogram')
            max_scenes: Maximum number of scenes to detect
            filter_shorter_than: Filter scenes shorter than duration (e.g., "2s", "30")

        Returns:
            List of paths to split scene files
        """
        from scripts.split_scenes import DetectorType, detect_and_split_scenes

        output_dir.mkdir(parents=True, exist_ok=True)

        detector_map = {
            "content": DetectorType.CONTENT,
            "adaptive": DetectorType.ADAPTIVE,
            "threshold": DetectorType.THRESHOLD,
            "histogram": DetectorType.HISTOGRAM,
        }

        detector = detector_map.get(detector_type.lower(), DetectorType.CONTENT)

        detect_and_split_scenes(
            video_path=str(video_path),
            output_dir=output_dir,
            detector_type=detector,
            threshold=threshold,
            min_scene_len=min_scene_length,
            max_scenes=max_scenes,
            filter_shorter_than=filter_shorter_than,
            save_images_per_scene=0,
        )

        scene_files = sorted(output_dir.glob(f"{video_path.stem}-Scene-*.mp4"))
        return scene_files

    def delete_dataset(self, dataset_name: str) -> None:
        """Delete entire dataset.

        Args:
            dataset_name: Name of the dataset to delete
        """
        dataset_dir = self.datasets_root / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
