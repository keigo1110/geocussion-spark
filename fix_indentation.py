with open('demo_collision_detection.py', 'r') as f:
    content = f.read()

# 39行目のfrom文のインデントを修正
content = content.replace(
    'try:\nfrom pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBError',
    'try:\n    from pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBError'
)

with open('demo_collision_detection.py', 'w') as f:
    f.write(content)
    
print('Fixed try block indentation')
