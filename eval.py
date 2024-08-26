
from preprocess import *
from ultralytics import YOLO


model = YOLO('yolov8n.pt')
model.train(data=data_yaml_path, epochs=20)
results = model.val()
# Save the trained model
model.save('/Users/lecourtoistristan/Documents/projet_IA/Mars_Crater_detection/best_model.pt')

def visualize_detections(model, image_path, n_samples=10):
    image_files = os.listdir(image_path)[:n_samples]
    for img_file in image_files:
        img_path = os.path.join(image_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_path)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        
        for result in results[0].boxes:
            x_min, y_min, x_max, y_max = result.xyxy[0].tolist()
            conf = result.conf[0].item()
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min, y_min, f'{conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.title(f'Detection in: {img_file}')
        plt.axis('off')
        plt.show()



if __name__ == "__main__":
    visualize_detections(model, test_img_path)