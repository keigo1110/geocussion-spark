with open('demo_collision_detection.py', 'r') as f:
    lines = f.readlines()

# 修正版を作成
new_lines = []
import_block = []
collecting_imports = False

for i, line in enumerate(lines):
    line_num = i + 1
    
    # 58行目 (pass) の後からimportブロックを検出
    if line_num == 59 and line.strip() == '':
        new_lines.append(line)
        collecting_imports = True
        continue
    
    # import部分を集める（8スペースから4スペースに変更）
    if collecting_imports and (line.startswith('        #') or line.startswith('        from') or line.startswith('        import') or line.startswith('        try:') or line.startswith('            from') or line.startswith('            print') or line.startswith('        except')):
        # インデントを調整（8スペース→4スペース）
        fixed_line = line.replace('        ', '    ', 1)  # 最初の8スペースを4スペースに
        import_block.append(fixed_line)
        continue
    
    # import収集終了を検出
    if collecting_imports and (line.strip() == '' or line.startswith('except ImportError')):
        # import_blockの内容を追加
        new_lines.extend(import_block)
        collecting_imports = False
        import_block = []
    
    new_lines.append(line)

# ファイルに書き戻し
with open('demo_collision_detection.py', 'w') as f:
    f.writelines(new_lines)

print('Fixed indentation for import statements')
