import cv2

img = cv2.imread('display/controller2.jpg')
alpha = 0.5
button_radius = 12
pressed_color = (0, 255, 0, 125)
X_center = (236, 123)
Y_center = (259, 102)
A_center = (258, 146)
B_center = (280, 124)

right_horizontal_from = (99, 123)
right_horizontal_to = (107, 123)
right_vertical_from = (103, 119)
right_vertical_to = (103, 127)

left_horizontal_from = (76, 123)
left_horizontal_to = (84, 123)
left_vertical_from = (80, 119)
left_vertical_to = (80, 127)

up_horizontal_from = (88, 112)
up_horizontal_to = (96, 112)
up_vertical_from = (92, 108)
up_vertical_to = (92, 116)

down_horizontal_from = (88, 132)
down_horizontal_to = (96, 132)
down_vertical_from = (92, 128)
down_vertical_to = (92, 136)


def get_controller_image(movement_index, action_index):
    overlay = img.copy()
    output = img.copy()

    if action_index == 0:
        cv2.circle(overlay, B_center, button_radius, pressed_color, -1)
    elif action_index == 1:
        cv2.circle(overlay, A_center, button_radius, pressed_color, -1)
    elif action_index == 2:
        cv2.circle(overlay, Y_center, button_radius, pressed_color, -1)
    elif action_index == 3:
        cv2.circle(overlay, X_center, button_radius, pressed_color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    if movement_index == 0:
        cv2.line(output, up_horizontal_from, up_horizontal_to, pressed_color, 2)
        cv2.line(output, up_vertical_from, up_vertical_to, pressed_color, 2)
    elif movement_index == 1:
        cv2.line(output, down_horizontal_from, down_horizontal_to, pressed_color, 2)
        cv2.line(output, down_vertical_from, down_vertical_to, pressed_color, 2)
    elif movement_index == 2:
        cv2.line(output, left_horizontal_from, left_horizontal_to, pressed_color, 2)
        cv2.line(output, left_vertical_from, left_vertical_to, pressed_color, 2)
    elif movement_index == 3:
        cv2.line(output, right_horizontal_from, right_horizontal_to, pressed_color, 2)
        cv2.line(output, right_vertical_from, right_vertical_to, pressed_color, 2)

    return output

# image_controller = get_controller_image(1, 1)
# cv2.imshow('controller', image_controller)
# cv2.waitKey(0)