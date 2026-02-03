import sqlite3
import sys

db_path = r'C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\test_20260116_134059\colmap_alignment\database.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check keypoints per image
cursor.execute('SELECT image_id, COUNT(*) as kp_count FROM keypoints GROUP BY image_id ORDER BY kp_count DESC LIMIT 10')
results = cursor.fetchall()
print('Top 10 images by keypoint count:')
for img_id, count in results:
    print(f'  Image {img_id}: {count} keypoints')

# Average keypoints
cursor.execute('SELECT COUNT(*) FROM keypoints')
total_kp = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(DISTINCT image_id) FROM keypoints')
num_images = cursor.fetchone()[0]

if num_images > 0:
    avg = total_kp / num_images
    print(f'\nTotal: {total_kp} keypoints across {num_images} images')
    print(f'Average: {avg:.1f} keypoints/image')
    print(f'\nEXPECTED: 500-5000 keypoints per image for SIFT')
    print(f'ACTUAL: {avg:.1f} keypoints/image - {"TOO LOW!" if avg < 100 else "OK"}')
else:
    print('\nNo keypoints found in database!')

conn.close()
