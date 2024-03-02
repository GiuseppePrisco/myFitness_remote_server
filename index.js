const express = require('express');
const multer  = require('multer');
const upload = multer({ dest: 'uploads/' });
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const posedetection = require('@tensorflow-models/pose-detection');

const app = express();

let detector;

let startTime = Date.now();
posedetection.createDetector(posedetection.SupportedModels.MoveNet).then((createdDetector) => {
    detector = createdDetector;
    let endTime = Date.now();
    console.log(`Time taken to create detector: ${endTime - startTime} ms`);
});



app.get('/', (req, res) => res.send('Server developed to measure the Performance for the MACC project'));


app.post('/upload', upload.single('image'), async (req, res) => {
    console.log("--------------------------------------------------");
    console.log("Received a new image. Starting performance measurements...");
    if (!detector) {
        return res.status(503).send('Server is starting up. Please try again later.');
    }


    let startTime = Date.now();
    const image = fs.readFileSync(req.file.path);
    let endTime = Date.now();
    console.log(`Time taken to read file: ${endTime - startTime} ms`);

    startTime = Date.now();
    const decodedImage = tf.node.decodeImage(new Uint8Array(image), 3);
    endTime = Date.now();
    console.log(`Time taken to decode image: ${endTime - startTime} ms`);

    startTime = Date.now();
    const poses = await detector.estimatePoses(decodedImage);
    endTime = Date.now();
    console.log(`Time taken to estimate poses: ${endTime - startTime} ms`);
    //console.dir(poses, { depth: null });
    poses.forEach((pose, index) => {
        console.log(`Pose ${index + 1} has ${pose.keypoints.length} landmarks.`);
    });

    startTime = Date.now();
    const mlKitPoses = poses.map(pose => ({
        poseLandmarks: pose.keypoints.map(keypoint => ({
            type: keypoint.name,
            position: {
                x: keypoint.x,
                y: keypoint.y,
                z: keypoint.score  // Assuming score is the z-coordinate
            },
            inFrameLikelihood: keypoint.score
        })),
        zRotation: 0  // You might need to calculate this value based on your keypoints
    }));
    // const mlKitPoses = poses.map(pose => ({
    //     poseLandmarks: pose.keypoints.map(keypoint => ({
    //         type: keypoint.name,
    //         position: {
    //             x: keypoint.position.x,
    //             y: keypoint.position.y,
    //             z: keypoint.position.z
    //         },
    //         inFrameLikelihood: keypoint.score
    //     })),
    //     zRotation: 0  // You might need to calculate this value based on your keypoints
    // }));
    endTime = Date.now();
    console.log(`Time taken to convert poses: ${endTime - startTime} ms`);


    res.send(mlKitPoses);
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
    console.log(`Server started on port ${port}`);
});

// app.listen(3000, () => {
//     console.log('Server started on port 3000');
// });
