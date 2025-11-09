import json

with open(r'e:\Deep Learning\Pytorch_Mastery\InceptionNet implementation.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total cells: {len(data['cells'])}")

for i, cell in enumerate(data['cells']):
    outputs = cell.get('outputs', [])
    if outputs:
        print(f"\nCell {i} ({cell.get('cell_type')}) has {len(outputs)} outputs")
        for j, output in enumerate(outputs):
            if 'text' in output:
                text = ''.join(output['text'])
                print(f"  Output {j}: {len(text)} chars")
                # Check for problematic characters
                for k, char in enumerate(text):
                    if 0xD800 <= ord(char) <= 0xDFFF or ord(char) == 0xDFAF:
                        print(f"    Found surrogate at position {k}: {hex(ord(char))}")
                if '✅' in text:
                    print(f"  Contains ✅ checkmark emoji")
