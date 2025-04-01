const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const { PythonShell } = require("python-shell");
const fs = require("fs");

const app = express();
const port = 5000;

// Middleware
app.use(cors()); 
app.use(express.json());

// Configure multer for handling file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, "uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB max file size
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith("image/")) {
      cb(null, true);
    } else {
      cb(new Error("Only image files are allowed"));
    }
  },
});

// Routes
app.post("/predict", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file provided" });
  }

  const region = req.body.region || "";
  const imagePath = req.file.path;

  // Options for PythonShell
  const options = {
    mode: "json",
    pythonPath: "python", // Adjust based on your system
    scriptPath: path.join(__dirname, "ml_model"),
    args: [imagePath, region],
  };

  // Call the Python script
  PythonShell.run("predict.py", options)
    .then((results) => {
      // Clean up the uploaded file
      fs.unlinkSync(imagePath);

      // Send results back to client
      if (results && results.length > 0) {
        res.json(results[0]);
      } else {
        res.status(500).json({ error: "No prediction results received" });
      }
    })
    .catch((err) => {
      console.error("Error during prediction:", err);

      // Clean up the uploaded file if it exists
      if (fs.existsSync(imagePath)) {
        fs.unlinkSync(imagePath);
      }

      res.status(500).json({
        error: "Failed to process the image",
        details: err.message,
      });
    });
});

// Check if model exists route
app.get("/model-status", (req, res) => {
  const modelPath = path.join(__dirname, "ml_model", "crop_model.joblib");
  const exists = fs.existsSync(modelPath);
  res.json({ modelExists: exists });
});

// Route to train the model
app.post("/train-model", (req, res) => {
  const options = {
    mode: "json",
    pythonPath: "python", // Adjust based on your system
    scriptPath: path.join(__dirname, "ml_model"),
  };

  PythonShell.run("train_model.py", options)
    .then((results) => {
      if (results && results.length > 0) {
        res.json(results[0]);
      } else {
        res.status(500).json({ error: "No training results received" });
      }
    })
    .catch((err) => {
      console.error("Error during model training:", err);
      res.status(500).json({
        error: "Failed to train the model",
        details: err.message,
      });
    });
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
