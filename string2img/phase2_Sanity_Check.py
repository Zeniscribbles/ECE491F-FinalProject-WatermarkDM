'''
@brief Phase 2 Sanity Check: Addressing the missing folder errors. 

Ensuring successful saves in Phase 2 to continue into Phase 3 without issue.
'''
import os

# Update these paths if you've changed the output structure
image_root = "../edm/datasets/embedded/cifar10/images"
note_root = "../edm/datasets/embedded/cifar10/note"

missing_notes = []
missing_images = []
total_images = 0
total_notes = 0

for i in range(50):
    folder = str(i).zfill(5)
    img_dir = os.path.join(image_root, folder)
    note_file = os.path.join(note_root, folder, "embedded_fingerprints.txt")

    # Check note file
    if not os.path.exists(note_file):
        missing_notes.append(folder)
    else:
        with open(note_file, "r") as f:
            lines = f.readlines()
            total_notes += len(lines)

    # Check if image dir exists and has PNGs
    if not os.path.isdir(img_dir) or not any(f.endswith('.png') for f in os.listdir(img_dir)):
        missing_images.append(folder)
    else:
        total_images += len([f for f in os.listdir(img_dir) if f.endswith('.png')])

print("\n Phase 2 Output Check Results")
print(f" Total embedded images found: {total_images}")
print(f" Total fingerprints (note entries): {total_notes}")
print(f" Missing image folders: {missing_images}")
print(f" Missing note files: {missing_notes}")
print(f" Ready for Phase 3: {'Yes' if not missing_images and not missing_notes else 'No'}")
