from pathlib import Path
import shutil


def save_uploaded_files(uploaded_files, target_dir):
    """Save uploaded file objects or local file paths into target_dir."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files or []:
        if isinstance(uploaded_file, (str, Path)):
            source = Path(uploaded_file)
            destination = target_dir / source.name
            shutil.copy2(source, destination)
        else:
            name = Path(uploaded_file.name).name
            destination = target_dir / name
            data = (
                uploaded_file.getbuffer()
                if hasattr(uploaded_file, "getbuffer")
                else uploaded_file.read()
            )
            if isinstance(data, str):
                data = data.encode("utf-8")
            destination.write_bytes(bytes(data))

        saved_paths.append(destination)

    return saved_paths
