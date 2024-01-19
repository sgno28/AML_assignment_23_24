import os
import tensorflow as tf
import numpy as np
import subprocess

def run_notebook(script_path):
    subprocess.run(['python', script_path])

def load_model_and_predict(model_path, test_image, test_labels):
    model = tf.keras.models.load_model(model_path)
    accuracy = model.evaluate(test_image, test_labels)
    return accuracy

def task_a_from_scratch():
    run_notebook('A/task_a.py')

def task_b_from_scratch():
    run_notebook('B/task_b.py')

def extract_test_data_and_conv(test_images, test_labels, mode):
    test_images = test_images.astype('float32') / 255
    if mode == 1: 
        test_images = np.expand_dims(test_images, axis=-1)  # Add an extra dimension for the channel
    test_labels = tf.keras.utils.to_categorical(test_labels)
    return test_images, test_labels

def task_a_from_final_model():
    data = np.load('Datasets/pneumoniamnist.npz')
    test_images = data['test_images']
    test_labels = data['test_labels']
    images, labels = extract_test_data_and_conv(test_images, test_labels, 1)
    acc = load_model_and_predict('A/balanced_model_91', images, labels)
    print("Acc for Task A from latest saved model:", acc)

def task_b_from_final_model():
    data = np.load('Datasets/pathmnist.npz')
    test_images = data['test_images']
    test_labels = data['test_labels']
    images, labels = extract_test_data_and_conv(test_images, test_labels, 0)

    acc = load_model_and_predict('B/taskB_model', images, labels)
    print("Acc for Task B from latest saved model:", acc)

def main():
    print("1. Run task A from scratch")
    print("2. Run task B from scratch")
    print("3. Run task A from latest saved model (testing set only)")
    print("4. Run task B from latest saved model (testing set only)")

    choice = input("Please select an option (1-4): ")

    if choice == '1':
        task_a_from_scratch()
    elif choice == '2':
        task_b_from_scratch()
    elif choice == '3':
        task_a_from_final_model()
    elif choice == '4':
        task_b_from_final_model()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()