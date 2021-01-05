import cv2
import matplotlib.pyplot as plt
import keras
import numpy as np

try:
    model = keras.models.load_model('myModel.h5', compile=False)

except Exception as e:
    print('Model couldn\'t be loaded', e)

def sort_contours(countours, method="left-to-right"): # We want to sort it left-to-right in this case

    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
# the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to bottom

    boundingBoxes = [cv2.boundingRect(c) for c in countours]
    (countours, boundingBoxes) = zip(*sorted(zip(countours, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes

    return (countours, boundingBoxes)


image = cv2.imread('finaltest.jpg')
image = cv2.resize(image, (700, 500), interpolation=cv2.INTER_AREA)


grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_contours = sort_contours(contours)
total_img = []
sortedlist_contours = []

j = 0
new_list = []
for c in sorted_contours[0]:
    # handles 2 lines at a time
    if j < 2:
        x, y, w, h = cv2.boundingRect(c)

        total_img.append(image[y-20:y+h+20, :])
        j = j + 1


thresh = [None] * 50
preprocessed_digits = []
for i in range(j):
    grey = cv2.cvtColor(total_img[i].copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh[i] = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sortedlist_contours.append(sort_contours(contours))


for i in range(j):
    for c in sortedlist_contours[i][0]:
        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(total_img[i], (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

        digit = thresh[i][y:y + h, x:x + w]

        resized_digit = cv2.resize(digit, (18, 18))

        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        preprocessed_digits.append(padded_digit)
        # plt.imshow(total_img[i], cmap="gray")
        # plt.show()


print("Final Image: ")
plt.imshow(image, cmap="gray")
plt.show()
inp = np.array(preprocessed_digits)

digits = []
digits1 = []
counter = 0
for digit in preprocessed_digits:
    prediction = model.predict(digit.reshape(1, 28, 28, 1))

    print("\n\n---------------------\n\n")
    print("PREDICTION")

    print(f"\nPrediction from the neural network in array:\n\n {prediction}")
    print(f"\n\nFinal Output: {np.argmax(prediction)}")


    if counter < 3:
        digits.append(np.argmax(prediction))
    else:
        digits1.append(np.argmax(prediction))
    counter = counter + 1

print("Prediction:", digits)
print("Prediction:", digits1)

sum1 = 0
for s in digits:
    sum1 = sum1 + s
    sum1 = sum1*10
print("Predicted top number", sum1/10)
sum2 = 0
for s1 in digits1:
    sum2 = sum2 + s1
    sum2 = sum2*10
print("Predicted bottom number", sum2/10)

while(1):
    answer = input("Would you like to subtract or add (Enter s or a respectively): ")
    if answer == 's' or answer == 'a':
        break

if answer == 's':
    difference = (sum1 - sum2)
    print("Subtracting...", difference)
else:
    Addition = (sum1 + sum2)
    print("Adding...", Addition)
