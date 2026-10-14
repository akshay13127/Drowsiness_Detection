# Driver Drowsiness Detection

Watches a driver through a webcam and sounds an alarm when their eyes stay closed too long. The model takes 24x24 eye crops, so it's light enough for a Raspberry Pi 3.

Python, OpenCV, Keras, pygame.

## How it works

1. **Find the face and eyes.** Each webcam frame goes to grayscale, and OpenCV Haar cascades locate the face, the left eye and the right eye.
2. **Classify each eye.** Both eye crops are resized to 24x24 grayscale and passed to a small CNN that labels them open or closed.
3. **Keep a running score.** Every frame with both eyes closed adds 1, every open frame subtracts 1, floored at 0. Blinks never get far.
4. **Raise the alarm.** Once the score passes 15, `alarm.wav` plays, a red border flashes on screen, and the frame is saved as `image.jpg`.

Press `q` to quit.

## The model

Defined and trained in `model.py`, saved to `models/cnnCat2.h5`.

```
Input 24x24x1
Conv2D 32 -> Conv2D 32 -> Conv2D 64   (3x3, ReLU)
Dropout -> Flatten -> Dense 128 -> Dropout
Dense 2, softmax   (open / closed)
```

It trains on grayscale eye crops arranged as `data/train/<class>/` and `data/valid/<class>/`. The dataset isn't in this repo.

## Run it

```bash
pip install opencv-python keras tensorflow pygame numpy
python "drowsiness detection.py"
```

The cascade paths in the script use Windows backslashes (`haar cascade files\...`). On Linux or macOS, switch them to forward slashes. Also note the script loads `models/cnncat2.h5` while the file is `cnnCat2.h5`, which breaks on case-sensitive filesystems like Linux.

## Files

| File | What it is |
| --- | --- |
| `drowsiness detection.py` | Webcam loop, detection and alarm |
| `model.py` | CNN definition and training |
| `models/cnnCat2.h5` | Trained weights |
| `haar cascade files/` | OpenCV face and eye cascades |
| `alarm.wav` | Alarm sound |
