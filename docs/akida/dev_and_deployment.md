### **Comprehensive Guide: How to Deploy an SNN**

Based on the **BrainChip documentation** you provided and the **Akida Evaluation Kit User Guide**, here is the workflow to prepare, transfer, and deploy a Spiking Neural Network (SNN) to your device.

#### **Phase 1: Prepare the Model (Host PC)**

*Note: You typically perform these steps on a more powerful computer (Host PC), not the Akida board itself, although the board is capable of running these tools if needed.*

1. **Create/Train a Model:**
You start with a standard TensorFlow/Keras model. You can train your own or use a pre-trained one from the **Akida Model Zoo** (`akida-models`).
* *Reference:* "1. Create and train".


2. **Quantize the Model (`quantizeml`):**
Standard neural networks use 32-bit floats. Akida requires reduced precision (typically 4-bit weights/activations). Use the `quantizeml` toolkit to convert your Keras model.
```python
import quantizeml
# Load your keras model
qmodel = quantizeml.models.quantize(keras_model)
# Fine-tune/calibrate if necessary

```


3. **Convert to Akida (`cnn2snn`):**
Convert the quantized model into the specific Akida binary format using the `cnn2snn` toolkit.
```python
from cnn2snn import convert
# Convert quantized model to Akida model
akida_model = convert(qmodel)
# Save the model for deployment
akida_model.save("my_snn_model.fbz")

```


* *Reference:* "3. Convert".





#### **Phase 2: Transfer Files ("How do I put it there?")**

You need to move your saved model (`.fbz`) and your inference Python script (`inference.py`) to the board.

**Option A: Secure Copy (SCP) - Recommended**
Run this command from your computer's terminal (Windows PowerShell or Mac/Linux Terminal):

```bash
scp my_snn_model.fbz bcdev@10.42.0.1:/home/bcdev/
scp inference.py bcdev@10.42.0.1:/home/bcdev/

```

*(Use password `Demo123` when prompted)*.

**Option B: USB Drive**
The board has USB ports. You can copy files to a USB stick, plug it into the Akida board, and copy them via the command line once logged in.



#### **Phase 3: Run Inference (On the Device)**

1. **Connect via SSH:**
Open your terminal/command prompt and log in to the board.
* **Command:** `ssh bcdev@10.42.0.1`
* **Password:** `Demo123`
* *(Note: Ensure you are connected to the Wi-Fi SSID `akida-devkit-CBD7`)*.


2. **Activate the Environment:**
You must activate the pre-installed Akida environment to access the software libraries.
* **Command:** `source venv_akida/bin/activate`
* *Reference:*.


3. **Run Your Code:**
Execute your Python script that loads the model and runs it on the hardware.
* **Command:** `python inference.py`



**Example Inference Code (`inference.py`):**
This is what your script on the board should look like to "run the inference code":

```python
import akida
import numpy as np

# 1. Load the model you transferred
model = akida.Model("my_snn_model.fbz")

# 2. Map to the hardware (The Akida NSoC on the board)
device = akida.Device()
model.map(device)

# 3. Prepare input data (Example: random data or image)
# (In a real scenario, you would load an image from the USB camera here)
input_data = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)

# 4. Run Inference
results = model.predict(input_data)

print("Inference results:", results)

```