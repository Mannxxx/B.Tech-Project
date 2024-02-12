# Open a CSV file for writing the corner coordinates
csv_file_path = 'corner_coordinates.csv'  # Specify the path to the CSV file
csv_file = open(csv_file_path, 'w', newline='')  # Open in 'write' mode
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'X', 'Y'])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    quality = cv2.getTrackbarPos("quality", "Frame") / 100
    max_corners = cv2.getTrackbarPos("max_corners", "Frame")

    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, 10)  # Adjust minDistance

    if corners is not None:
        corners = np.int0(corners)

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # Write corner coordinates to CSV
            csv_writer.writerow([frame_count, x, y])

    # Extract minimum and maximum coordinates from the CSV and mark them on the image
    min_row, max_row = extract_extremes(csv_file_path, column_index=1)  # Use the correct file path here
    if min_row:
        min_x, min_y = int(min_row[1]), int(min_row[2])
        cv2.circle(frame, (min_x, min_y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f'Min: ({min_x}, {min_y})', (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if max_row:
        max_x, max_y = int(max_row[1]), int(max_row[2])
        cv2.circle(frame, (max_x, max_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, f'Max: ({max_x}, {max_y})', (max_x, max_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    frame_count += 1

# Close the CSV file
csv_file.close()

cap.release()
cv2.destroyAllWindows()
