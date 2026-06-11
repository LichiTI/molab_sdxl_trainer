"""Directory strategy and listing helpers for legacy picker routes."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


PICKER_FILE_EXTENSIONS = {
    "model-file": {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".yaml", ".json"},
    "output-model-file": {".safetensors", ".ckpt", ".pt", ".pth"},
    "model-saved-file": {".safetensors", ".ckpt", ".pt", ".pth"},
    "image-file": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"},
    "text-file": {".txt", ".caption", ".json", ".jsonl", ".toml", ".yaml", ".yml"},
    "file": {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".yaml", ".json"},
}


def resolve_picker_kind(pick_type: str, *, context: str = "", project_root: Path, backend_root: Path) -> tuple[Path, str, bool]:
    normalized_type = str(pick_type or "").strip().lower()
    ctx = str(context or "").strip().lower().replace("-", "_")
    model_root = _first_existing(project_root / "sd-models", project_root / "models", backend_root / "models")
    train_root = _first_existing(project_root / "sucai", project_root / "train", project_root / "data", backend_root / "data")
    output_root = project_root / "output"
    logs_root = project_root / "logs"
    model_dir_contexts = {
        "pretrained_model_name_or_path",
        "transformer_path",
        "gemma_model_path",
        "clip_model_path",
        "vae_path",
        "t5_tokenizer_path",
    }
    output_contexts = {"output_dir", "resume", "resize_output"}
    log_contexts = {"logging_dir"}
    if normalized_type in {"model-file", "file"}:
        return model_root, "file", True
    if normalized_type in {"output-model-file", "model-saved-file"}:
        return output_root, "file", True
    if normalized_type in {"image-file", "text-file"}:
        return train_root, "file", True
    if normalized_type == "output-folder":
        return output_root, "folder", False
    if normalized_type == "train-dir":
        return train_root, "folder", False
    if normalized_type == "folder":
        if ctx in model_dir_contexts or "model" in ctx or "tokenizer" in ctx or "transformer" in ctx:
            return model_root, "folder", False
        if ctx in output_contexts or ctx.startswith("output") or "output" in ctx or "resume" in ctx:
            return output_root, "folder", False
        if ctx in log_contexts or "log" in ctx:
            return logs_root, "folder", False
        return train_root, "folder", False
    return project_root, "folder", False


def list_picker_entries(pick_type: str, *, context: str = "", project_root: Path, backend_root: Path) -> list[dict[str, Any]]:
    root, kind, recursive = resolve_picker_kind(pick_type, context=context, project_root=project_root, backend_root=backend_root)
    if not root.is_dir():
        return []
    entries = []
    allowed_exts = PICKER_FILE_EXTENSIONS.get(str(pick_type or "").strip().lower())
    iterator = root.rglob("*") if recursive else root.iterdir()
    try:
        for entry in sorted(iterator):
            if entry.name.startswith("."):
                continue
            if kind == "folder" and not entry.is_dir():
                continue
            if kind == "file":
                if not entry.is_file():
                    continue
                if allowed_exts and entry.suffix.lower() not in allowed_exts:
                    continue
            entries.append({"name": entry.name, "path": str(entry), "is_dir": entry.is_dir()})
    except (PermissionError, OSError):
        return entries
    return entries


def list_pick_files(pick_type: str, *, project_root: Path, backend_root: Path) -> list[dict[str, Any]]:
    return list_picker_entries(pick_type, context="", project_root=project_root, backend_root=backend_root)


def build_builtin_picker_payload(picker_type: str, *, context: str = "", project_root: Path, backend_root: Path) -> dict[str, Any]:
    root, _kind, _recursive = resolve_picker_kind(picker_type, context=context, project_root=project_root, backend_root=backend_root)
    files = list_picker_entries(picker_type, context=context, project_root=project_root, backend_root=backend_root)
    root_label = str(root).replace("\\", "/") if root else ""
    prefix = root_label + "/" if root_label else ""
    items = []
    for entry in files:
        path = str(entry["path"]).replace("\\", "/")
        items.append(path[len(prefix):] if prefix and path.startswith(prefix) else entry["name"])
    return {"rootLabel": root_label, "items": items}


def build_pick_file_payload(
    picker_type: str,
    *,
    context: str = "",
    project_root: Path,
    backend_root: Path,
    open_picker_fn: Any = None,
) -> dict[str, str]:
    root, kind, _recursive = resolve_picker_kind(
        picker_type,
        context=context,
        project_root=project_root,
        backend_root=backend_root,
    )
    initial_dir = root if root.exists() else project_root
    picker = open_picker_fn or open_native_path_picker
    selected = picker(picker_type=picker_type, kind=kind, initial_dir=initial_dir)
    return {"path": str(selected or "")}


def build_get_files_payload(pick_type: str, *, project_root: Path, backend_root: Path) -> dict[str, Any]:
    return {"files": list_pick_files(pick_type, project_root=project_root, backend_root=backend_root)}


def build_pick_file_route_payload(
    picker_type: str,
    *,
    context: str = "",
    project_root: Path,
    backend_root: Path,
    open_picker_fn: Any = None,
) -> dict[str, str]:
    return build_pick_file_payload(
        picker_type,
        context=context,
        project_root=project_root,
        backend_root=backend_root,
        open_picker_fn=open_picker_fn,
    )


def build_get_files_route_payload(pick_type: str, *, project_root: Path, backend_root: Path) -> dict[str, Any]:
    return build_get_files_payload(pick_type, project_root=project_root, backend_root=backend_root)


def build_builtin_picker_route_payload(
    picker_type: str,
    *,
    context: str = "",
    project_root: Path,
    backend_root: Path,
) -> dict[str, Any]:
    return build_builtin_picker_payload(
        picker_type,
        context=context,
        project_root=project_root,
        backend_root=backend_root,
    )


def picker_filetypes(picker_type: str) -> list[tuple[str, str]]:
    normalized = str(picker_type or "").strip().lower()
    if normalized in {"model-file", "file"}:
        return [
            ("Model files", "*.safetensors *.ckpt *.pt *.pth *.bin *.gguf"),
            ("Config files", "*.yaml *.yml *.json *.toml"),
            ("All files", "*.*"),
        ]
    if normalized in {"output-model-file", "model-saved-file"}:
        return [("Model files", "*.safetensors *.ckpt *.pt *.pth"), ("All files", "*.*")]
    if normalized == "image-file":
        return [("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.gif"), ("All files", "*.*")]
    if normalized == "text-file":
        return [("Text/config files", "*.txt *.caption *.json *.jsonl *.toml *.yaml *.yml"), ("All files", "*.*")]
    return [("All files", "*.*")]


def open_native_path_picker(*, picker_type: str, kind: str, initial_dir: Path) -> str:
    """Open the OS-native file/folder picker.

    On Windows we go straight to PowerShell + WinForms. Tk can deadlock when
    it is created from a FastAPI worker thread instead of the process main
    thread, which looks to users like the picker froze the whole app.
    """
    if sys.platform == "win32":
        return open_native_path_picker_powershell(picker_type=picker_type, kind=kind, initial_dir=initial_dir)
    selected = open_native_path_picker_tk(picker_type=picker_type, kind=kind, initial_dir=initial_dir)
    return selected or ""


def open_native_path_picker_tk(*, picker_type: str, kind: str, initial_dir: Path) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        if kind == "folder":
            value = filedialog.askdirectory(
                title="请选择目录",
                initialdir=str(initial_dir),
                mustexist=False,
            )
        else:
            value = filedialog.askopenfilename(
                title="请选择文件",
                initialdir=str(initial_dir),
                filetypes=picker_filetypes(picker_type),
            )
        root.destroy()
        return str(value or "")
    except Exception as exc:
        logger.debug("tkinter picker failed: %s", exc)
        return None


def _powershell_filter_for(picker_type: str) -> str:
    """Build a WinForms-compatible filter string from picker_filetypes."""
    parts: list[str] = []
    for label, patterns in picker_filetypes(picker_type):
        winforms_patterns = patterns.strip().replace(" ", ";")
        parts.append(label.replace("|", "/"))
        parts.append(winforms_patterns or "*.*")
    return "|".join(parts) if parts else "All files (*.*)|*.*"


def open_native_path_picker_powershell(*, picker_type: str, kind: str, initial_dir: Path) -> str:
    """Open the modern Win10/11 file/folder dialog via PowerShell + WinForms."""
    escaped_dir = str(initial_dir).replace("'", "''")

    # Build an invisible top-most owner so the dialog cannot hide behind the UI.
    common_prelude = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "Add-Type -AssemblyName System.Drawing; "
        "$nativePickerWin32Src = '"
        "using System; "
        "using System.IO; "
        "using System.Runtime.InteropServices; "
        "public static class NativePickerWin32 { "
        "  [DllImport(\"user32.dll\")] public static extern bool SetForegroundWindow(IntPtr hWnd); "
        "  [DllImport(\"user32.dll\")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow); "
        "  [DllImport(\"user32.dll\")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags); "
        "} "
        "[ComImport, Guid(\"DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7\")] public class FileOpenDialog { } "
        "[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid(\"43826D1E-E718-42EE-BC55-A1E261C37BFE\")] public interface IShellItem { void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv); void GetParent(out IShellItem ppsi); void GetDisplayName(uint sigdnName, out IntPtr ppszName); void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs); void Compare(IShellItem psi, uint hint, out int piOrder); } "
        "[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid(\"42F85136-DB7E-439C-85F1-E4075D135FC8\")] public interface IFileDialog { [PreserveSig] int Show(IntPtr parent); void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec); void SetFileTypeIndex(uint iFileType); void GetFileTypeIndex(out uint piFileType); void Advise(IntPtr pfde, out uint pdwCookie); void Unadvise(uint dwCookie); void SetOptions(uint fos); void GetOptions(out uint fos); void SetDefaultFolder(IShellItem psi); void SetFolder(IShellItem psi); void GetFolder(out IShellItem ppsi); void GetCurrentSelection(out IShellItem ppsi); void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName); void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName); void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle); void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText); void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel); void GetResult(out IShellItem ppsi); void AddPlace(IShellItem psi, uint fdap); void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension); void Close(int hr); void SetClientGuid(ref Guid guid); void ClearClientData(); void SetFilter(IntPtr pFilter); } "
        "[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid(\"D57C7288-D4AD-4768-BE02-9D969532D960\")] public interface IFileOpenDialog : IFileDialog { void GetResults(out IntPtr ppenum); void GetSelectedItems(out IntPtr ppsai); } "
        "public static class NativeFolderPicker { [DllImport(\"shell32.dll\", CharSet=CharSet.Unicode, PreserveSig=false)] public static extern void SHCreateItemFromParsingName(string pszPath, IntPtr pbc, ref Guid riid, out IShellItem ppv); public static string PickFolder(IntPtr owner, string initialDir) { IFileOpenDialog dialog = (IFileOpenDialog)new FileOpenDialog(); uint options; dialog.GetOptions(out options); dialog.SetOptions(options | 0x20 | 0x40 | 0x800); dialog.SetTitle(\"请选择文件夹\"); dialog.SetOkButtonLabel(\"选择文件夹\"); if (!String.IsNullOrWhiteSpace(initialDir) && Directory.Exists(initialDir)) { try { Guid shellItemGuid = new Guid(\"43826D1E-E718-42EE-BC55-A1E261C37BFE\"); IShellItem folderItem; SHCreateItemFromParsingName(initialDir, IntPtr.Zero, ref shellItemGuid, out folderItem); dialog.SetFolder(folderItem); } catch { } } int hr = dialog.Show(owner); if (hr != 0) return String.Empty; IShellItem result; dialog.GetResult(out result); IntPtr pathPtr; result.GetDisplayName(0x80058000, out pathPtr); try { return Marshal.PtrToStringUni(pathPtr) ?? String.Empty; } finally { Marshal.FreeCoTaskMem(pathPtr); } } } "
        "'; Add-Type -TypeDefinition $nativePickerWin32Src; "
        "$wa = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea; "
        "$owner = New-Object System.Windows.Forms.Form; "
        "$owner.StartPosition = 'Manual'; "
        "$owner.Location = New-Object System.Drawing.Point([int]($wa.Left + $wa.Width / 2), [int]($wa.Top + $wa.Height / 2)); "
        "$owner.Size = New-Object System.Drawing.Size(1, 1); "
        "$owner.ShowInTaskbar = $false; "
        "$owner.FormBorderStyle = 'FixedToolWindow'; "
        "$owner.Opacity = 0.01; "
        "$owner.TopMost = $true; "
        "$owner.Show(); "
        "$owner.BringToFront(); "
        "$owner.Activate(); "
        "$HWND_TOPMOST = [IntPtr](-1); "
        "$SW_SHOWNORMAL = 1; "
        "$SWP_NOSIZE = 0x0001; $SWP_NOMOVE = 0x0002; $SWP_SHOWWINDOW = 0x0040; "
        "[NativePickerWin32]::ShowWindow($owner.Handle, $SW_SHOWNORMAL) | Out-Null; "
        "[NativePickerWin32]::SetWindowPos($owner.Handle, $HWND_TOPMOST, 0, 0, 0, 0, $SWP_NOMOVE -bor $SWP_NOSIZE -bor $SWP_SHOWWINDOW) | Out-Null; "
        "[NativePickerWin32]::SetForegroundWindow($owner.Handle) | Out-Null; "
        "$owner.Activate(); "
    )

    if kind == "folder":
        ps = (
            common_prelude
            + f"$selectedPath = [NativeFolderPicker]::PickFolder($owner.Handle, '{escaped_dir}'); "
            + "$owner.Close(); "
            + "if (![string]::IsNullOrWhiteSpace($selectedPath)) "
            + "{ [Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); Write-Output $selectedPath }"
        )
    else:
        filter_str = _powershell_filter_for(picker_type).replace("'", "''")
        ps = (
            common_prelude
            + "$d = New-Object System.Windows.Forms.OpenFileDialog; "
            + f"$d.InitialDirectory = '{escaped_dir}'; "
            + f"$d.Filter = '{filter_str}'; "
            + "$d.AutoUpgradeEnabled = $true; "
            + "$d.Title = '请选择文件'; "
            + "$d.CheckFileExists = $true; "
            + "$d.RestoreDirectory = $true; "
            + "$result = $d.ShowDialog($owner); "
            + "$owner.Close(); "
            + "if ($result -eq [System.Windows.Forms.DialogResult]::OK) "
            + "{ [Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); Write-Output $d.FileName }"
        )
    completed = subprocess.run(
        ["powershell", "-STA", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    if completed.returncode != 0:
        logger.warning(
            "PowerShell picker exited %s: stderr=%s",
            completed.returncode,
            (completed.stderr or "").strip(),
        )
        return ""
    return (completed.stdout or "").strip()


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]
