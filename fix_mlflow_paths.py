#!/usr/bin/env python3
"""
Script để sửa tất cả đường dẫn Windows tuyệt đối trong MLflow metadata
thành đường dẫn Linux tương đối trong Docker container.
"""
import os
import re
import sys

def fix_windows_paths(root_dir):
    """Sửa tất cả đường dẫn Windows trong MLflow metadata files."""
    fixed_count = 0
    target_base = '/app/mlruns'
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file in ['meta.yaml', 'MLmodel']:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Pattern 1: file:///C:/path/to/mlruns -> /app/mlruns
                    content = re.sub(
                        r'file:///[A-Z]:[/\\].*?[/\\]mlruns[/\\]?',
                        target_base,
                        content
                    )
                    
                    # Pattern 2: C:/path/to/mlruns -> /app/mlruns
                    content = re.sub(
                        r'[A-Z]:[/\\].*?[/\\]mlruns[/\\]?',
                        target_base,
                        content
                    )
                    
                    # Pattern 3: file:///C:\path\to\mlruns -> /app/mlruns (backslash)
                    content = re.sub(
                        r'file:///[A-Z]:[\\/].*?[\\/]mlruns[\\/]?',
                        target_base,
                        content
                    )
                    
                    # Pattern 4: Sửa storage_location trong meta.yaml
                    # storage_location: file:///C:/... -> /app/mlruns/...
                    if 'storage_location:' in content:
                        # Lấy phần sau storage_location và sửa
                        content = re.sub(
                            r'storage_location:\s*(file:///)?[A-Z]:[/\\].*?[/\\]mlruns[/\\](.*)',
                            lambda m: f'storage_location: {target_base}/{m.group(2) if len(m.groups()) > 1 and m.group(2) else ""}',
                            content
                        )
                    
                    # Pattern 5: Sửa artifact_path trong MLmodel
                    # artifact_path: file:///C:/... -> /app/mlruns/...
                    if 'artifact_path:' in content:
                        content = re.sub(
                            r'artifact_path:\s*(file:///)?[A-Z]:[/\\].*?[/\\]mlruns[/\\](.*)',
                            lambda m: f'artifact_path: {target_base}/{m.group(2) if len(m.groups()) > 1 and m.group(2) else ""}',
                            content
                        )
                    
                    # Ghi lại nếu có thay đổi
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixed_count += 1
                        print(f"[OK] Fixed: {filepath}")
                        
                except Exception as e:
                    print(f"[WARN] Error processing {filepath}: {e}")
                    continue
    
    return fixed_count

if __name__ == '__main__':
    mlruns_path = sys.argv[1] if len(sys.argv) > 1 else '/app/mlruns'
    print(f"[INFO] Fixing Windows paths in: {mlruns_path}")
    count = fix_windows_paths(mlruns_path)
    print(f"[OK] Fixed {count} files")
    sys.exit(0 if count > 0 or os.path.exists(mlruns_path) else 1)

