# 完全に新しいファイルを作成する
with open('demo_collision_detection.py.backup', 'r') as f:
    content = f.read()

# 問題のある部分を特定して修正
lines = content.split('\n')

# 新しいファイル内容を作成
new_lines = []
in_inner_except = False
inner_except_indent = 0

for i, line in enumerate(lines):
    # 内側のexcept ImportError:ブロック開始
    if line.strip() == 'except ImportError:' and '    try:' in lines[i-5:i]:
        new_lines.append(line)
        in_inner_except = True
        inner_except_indent = len(line) - len(line.lstrip())
        continue
    
    # 内側のexceptブロック内のimport文を移動
    if in_inner_except and line.startswith(' ' * (inner_except_indent + 4)):
        if 'Import OBFormat' in line:
            # ここから外側のtryブロックレベルにimport文を移動
            in_inner_except = False
            new_lines.append('')  # 空行追加
            new_lines.append('    # Import OBFormat from common types')
            continue
    
    # 内側のexceptブロック内の他の文も外側に移動
    if in_inner_except and (line.strip().startswith('from ') or line.strip().startswith('import ') or line.strip().startswith('#')):
        # インデントを外側のtryブロックレベルに調整
        fixed_line = '    ' + line.strip()
        new_lines.append(fixed_line)
        continue
    
    new_lines.append(line)

# 修正されたファイルを書き出し
with open('demo_collision_detection.py', 'w') as f:
    f.write('\n'.join(new_lines))

print('Created fixed demo file')
