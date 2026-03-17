/* BlazePose Fullbody Unit Tests */
/*
 * These tests verify that the C# BlazePose computational logic
 * produces results consistent with the Python BlazePose reference implementation.
 *
 * Python reference:
 *   - blazepose-fullbody/blazepose-fullbody.py
 *
 * Tests cover:
 *   - Sigmoid activation function
 *   - Box decoding and non-max suppression
 *   - Landmark decoding
 *   - GetResult (affine transform of landmarks to original image coordinates)
 *   - Box overlap (Jaccard/IoU)
 *   - Box merge (weighted non-max suppression)
 */

using NUnit.Framework;
using System;
using System.Collections.Generic;
using UnityEngine;

[TestFixture]
public class BlazePoseUnitTest
{
    private AiliaBlazepose blazepose = null!;
    private const float Tolerance = 1e-5f;

    [SetUp]
    public void SetUp()
    {
        blazepose = new AiliaBlazepose();
    }

    [TearDown]
    public void TearDown()
    {
        blazepose.Dispose();
    }

    // =======================================================
    // 1. Sigmoid
    //    Python: 1 / (1 + np.exp(-x))
    // =======================================================
    [Test]
    public void Sigmoid_Zero_Returns0_5()
    {
        Assert.That(blazepose.Sigmoid(0f), Is.EqualTo(0.5f).Within(Tolerance));
    }

    [Test]
    public void Sigmoid_LargePositive_ReturnsNear1()
    {
        float result = blazepose.Sigmoid(10f);
        Assert.That(result, Is.GreaterThan(0.999f));
        Assert.That(result, Is.LessThanOrEqualTo(1.0f));
    }

    [Test]
    public void Sigmoid_LargeNegative_ReturnsNear0()
    {
        float result = blazepose.Sigmoid(-10f);
        Assert.That(result, Is.LessThan(0.001f));
        Assert.That(result, Is.GreaterThanOrEqualTo(0.0f));
    }

    [Test]
    public void Sigmoid_MatchesPythonValues()
    {
        // Python: scipy.special.expit([−5, −1, 0, 1, 5])
        // → [0.00669285, 0.26894142, 0.5, 0.73105858, 0.99330715]
        Assert.That(blazepose.Sigmoid(-5f), Is.EqualTo(0.00669285f).Within(1e-4f));
        Assert.That(blazepose.Sigmoid(-1f), Is.EqualTo(0.26894142f).Within(1e-4f));
        Assert.That(blazepose.Sigmoid(1f), Is.EqualTo(0.73105858f).Within(1e-4f));
        Assert.That(blazepose.Sigmoid(5f), Is.EqualTo(0.99330715f).Within(1e-4f));
    }

    [Test]
    public void Sigmoid_Symmetry()
    {
        // sigmoid(x) + sigmoid(-x) = 1
        float[] values = { 0.5f, 1.0f, 2.5f, 7.0f };
        foreach (var x in values)
        {
            Assert.That(blazepose.Sigmoid(x) + blazepose.Sigmoid(-x),
                Is.EqualTo(1.0f).Within(Tolerance), $"sigmoid({x}) + sigmoid(-{x}) == 1");
        }
    }

    // =======================================================
    // 2. Box.GetJaccardOverlap (IoU)
    //    Python: intersection / union
    // =======================================================
    [Test]
    public void GetJaccardOverlap_IdenticalBoxes_Returns1()
    {
        var box = CreateBox(0, 0, 1, 1);
        Assert.That(box.GetJaccardOverlap(box), Is.EqualTo(1.0f).Within(Tolerance));
    }

    [Test]
    public void GetJaccardOverlap_NoOverlap_Returns0()
    {
        var box1 = CreateBox(0, 0, 1, 1);
        var box2 = CreateBox(2, 2, 3, 3);
        Assert.That(box1.GetJaccardOverlap(box2), Is.EqualTo(0.0f).Within(Tolerance));
    }

    [Test]
    public void GetJaccardOverlap_PartialOverlap()
    {
        var box1 = CreateBox(0, 0, 2, 2);  // area = 4
        var box2 = CreateBox(1, 1, 3, 3);  // area = 4
        // overlap = 1x1 = 1, union = 4 + 4 - 1 = 7
        Assert.That(box1.GetJaccardOverlap(box2), Is.EqualTo(1.0f / 7.0f).Within(Tolerance));
    }

    [Test]
    public void GetJaccardOverlap_ContainedBox()
    {
        var outer = CreateBox(0, 0, 4, 4);  // area = 16
        var inner = CreateBox(1, 1, 2, 2);  // area = 1
        // overlap = 1, union = 16 + 1 - 1 = 16
        Assert.That(outer.GetJaccardOverlap(inner), Is.EqualTo(1.0f / 16.0f).Within(Tolerance));
    }

    // =======================================================
    // 3. Box.Merge / Box.FinalizeMerge (weighted NMS)
    // =======================================================
    [Test]
    public void BoxMerge_TwoBoxes_WeightedAverage()
    {
        var box1 = CreateBox(0, 0, 2, 2, 0.8f);
        var box2 = CreateBox(1, 1, 3, 3, 0.2f);

        box1.Merge(box2);
        box1.FinalizeMerge();

        // Weighted average: (0.8*0 + 0.2*1) / (0.8+0.2) = 0.2
        Assert.That(box1.xMin, Is.EqualTo(0.2f).Within(Tolerance));
        Assert.That(box1.yMin, Is.EqualTo(0.2f).Within(Tolerance));
        Assert.That(box1.xMax, Is.EqualTo(2.2f).Within(Tolerance));
        Assert.That(box1.yMax, Is.EqualTo(2.2f).Within(Tolerance));
        // score = (0.8+0.2) / (1+1) = 0.5
        Assert.That(box1.score, Is.EqualTo(0.5f).Within(Tolerance));
    }

    [Test]
    public void BoxFinalizeMerge_NoMerge_NoChange()
    {
        var box = CreateBox(1, 2, 3, 4, 0.9f);
        box.FinalizeMerge();
        Assert.That(box.xMin, Is.EqualTo(1f).Within(Tolerance));
        Assert.That(box.yMin, Is.EqualTo(2f).Within(Tolerance));
        Assert.That(box.xMax, Is.EqualTo(3f).Within(Tolerance));
        Assert.That(box.yMax, Is.EqualTo(4f).Within(Tolerance));
        Assert.That(box.score, Is.EqualTo(0.9f).Within(Tolerance));
    }

    // =======================================================
    // 4. DecodeAndProcessBoxes
    //    Python: decode_boxes + weighted_nms
    // =======================================================
    [Test]
    public void DecodeAndProcessBoxes_AllLowScores_EmptyResult()
    {
        // All raw scores set to large negative -> sigmoid < threshold
        for (int i = 0; i < blazepose.rawScoresOutput.Length; i++)
            blazepose.rawScoresOutput[i] = -100f;

        blazepose.DecodeAndProcessBoxes();
        Assert.That(blazepose.boxes.Count, Is.EqualTo(0));
    }

    [Test]
    public void DecodeAndProcessBoxes_SingleHighScore_OneBox()
    {
        // Set all scores to very low
        for (int i = 0; i < blazepose.rawScoresOutput.Length; i++)
            blazepose.rawScoresOutput[i] = -100f;

        // Set one anchor with high score
        int tensorIdx = 100;
        blazepose.rawScoresOutput[tensorIdx] = 5f;  // sigmoid(5) ≈ 0.993

        // Set anchor center and size
        blazepose.anchors[tensorIdx, 0] = 0.5f;  // x center
        blazepose.anchors[tensorIdx, 1] = 0.5f;  // y center
        blazepose.anchors[tensorIdx, 2] = 1.0f;  // width scale
        blazepose.anchors[tensorIdx, 3] = 1.0f;  // height scale

        // Set raw box data (relative to anchor)
        int offset = (int)(tensorIdx * AiliaBlazepose.BLAZEPOSE_DETECTOR_KEYPOINT_COUNT * 3); // TENSOR_SIZE = 12
        blazepose.rawBoxesOutput[tensorIdx * 12 + 0] = 0;   // xCenter offset
        blazepose.rawBoxesOutput[tensorIdx * 12 + 1] = 0;   // yCenter offset
        blazepose.rawBoxesOutput[tensorIdx * 12 + 2] = 112;  // width
        blazepose.rawBoxesOutput[tensorIdx * 12 + 3] = 112;  // height

        // Set keypoints
        for (int k = 0; k < 4; k++)
        {
            blazepose.rawBoxesOutput[tensorIdx * 12 + 4 + k * 2] = 0;
            blazepose.rawBoxesOutput[tensorIdx * 12 + 4 + k * 2 + 1] = 0;
        }

        blazepose.DecodeAndProcessBoxes();
        Assert.That(blazepose.boxes.Count, Is.EqualTo(1));
        Assert.That(blazepose.boxes[0].score, Is.GreaterThan(0.5f));
    }

    [Test]
    public void DecodeAndProcessBoxes_OverlappingBoxes_Merged()
    {
        // Set all scores low
        for (int i = 0; i < blazepose.rawScoresOutput.Length; i++)
            blazepose.rawScoresOutput[i] = -100f;

        // Create two overlapping detections at nearby anchors
        int[] tensorIndices = { 100, 101 };
        float[] scores = { 4f, 3f };  // sigmoid ≈ 0.982, 0.953

        for (int t = 0; t < tensorIndices.Length; t++)
        {
            int ti = tensorIndices[t];
            blazepose.rawScoresOutput[ti] = scores[t];

            blazepose.anchors[ti, 0] = 0.5f;
            blazepose.anchors[ti, 1] = 0.5f;
            blazepose.anchors[ti, 2] = 1.0f;
            blazepose.anchors[ti, 3] = 1.0f;

            blazepose.rawBoxesOutput[ti * 12 + 0] = 0;
            blazepose.rawBoxesOutput[ti * 12 + 1] = 0;
            blazepose.rawBoxesOutput[ti * 12 + 2] = 112;
            blazepose.rawBoxesOutput[ti * 12 + 3] = 112;

            for (int k = 0; k < 4; k++)
            {
                blazepose.rawBoxesOutput[ti * 12 + 4 + k * 2] = 0;
                blazepose.rawBoxesOutput[ti * 12 + 4 + k * 2 + 1] = 0;
            }
        }

        blazepose.DecodeAndProcessBoxes();
        // Should merge into a single box since they overlap significantly
        Assert.That(blazepose.boxes.Count, Is.EqualTo(1));
    }

    // =======================================================
    // 5. DecodeAndProcessLandmarks
    //    Python: landmark decoding from estimation output
    // =======================================================
    [Test]
    public void DecodeAndProcessLandmarks_33Landmarks()
    {
        // Fill estimation output with known values
        // Each landmark has 5 values: x, y, z, visibility, presence
        int resolution = 256;
        for (int i = 0; i < 33; i++)
        {
            blazepose.estimationOutputBuffer[i * 5 + 0] = resolution * 0.5f;  // x = center
            blazepose.estimationOutputBuffer[i * 5 + 1] = resolution * 0.5f;  // y = center
            blazepose.estimationOutputBuffer[i * 5 + 2] = 0;                   // z = 0
            blazepose.estimationOutputBuffer[i * 5 + 3] = 5f;                  // visibility (high)
            blazepose.estimationOutputBuffer[i * 5 + 4] = 5f;                  // presence (high)
        }

        blazepose.DecodeAndProcessLandmarks();

        Assert.That(blazepose.landmarks.Count, Is.EqualTo(33));
        foreach (var lm in blazepose.landmarks)
        {
            Assert.That(lm.position.x, Is.EqualTo(0.5f).Within(Tolerance), "x normalized");
            Assert.That(lm.position.y, Is.EqualTo(0.5f).Within(Tolerance), "y normalized");
            Assert.That(lm.position.z, Is.EqualTo(0f).Within(Tolerance), "z");
            Assert.That(lm.confidence, Is.GreaterThan(0.9f), "high confidence");
        }
    }

    [Test]
    public void DecodeAndProcessLandmarks_LowConfidence()
    {
        for (int i = 0; i < 33; i++)
        {
            blazepose.estimationOutputBuffer[i * 5 + 0] = 128;
            blazepose.estimationOutputBuffer[i * 5 + 1] = 128;
            blazepose.estimationOutputBuffer[i * 5 + 2] = 0;
            blazepose.estimationOutputBuffer[i * 5 + 3] = -5f;  // low visibility
            blazepose.estimationOutputBuffer[i * 5 + 4] = -5f;  // low presence
        }

        blazepose.DecodeAndProcessLandmarks();

        Assert.That(blazepose.landmarks.Count, Is.EqualTo(33));
        foreach (var lm in blazepose.landmarks)
        {
            Assert.That(lm.confidence, Is.LessThan(0.1f), "low confidence");
        }
    }

    [Test]
    public void DecodeAndProcessLandmarks_ConfidenceUsesMinOfVisibilityPresence()
    {
        // Python: sigmoid(min(visibility, presence))
        blazepose.estimationOutputBuffer[0] = 128;  // x
        blazepose.estimationOutputBuffer[1] = 128;  // y
        blazepose.estimationOutputBuffer[2] = 0;    // z
        blazepose.estimationOutputBuffer[3] = 10f;  // visibility (very high)
        blazepose.estimationOutputBuffer[4] = -2f;  // presence (low)

        // Fill rest with defaults
        for (int i = 1; i < 33; i++)
        {
            blazepose.estimationOutputBuffer[i * 5 + 0] = 128;
            blazepose.estimationOutputBuffer[i * 5 + 1] = 128;
            blazepose.estimationOutputBuffer[i * 5 + 2] = 0;
            blazepose.estimationOutputBuffer[i * 5 + 3] = 0;
            blazepose.estimationOutputBuffer[i * 5 + 4] = 0;
        }

        blazepose.DecodeAndProcessLandmarks();

        // Confidence should use min(-2, 10) = -2 -> sigmoid(-2) ≈ 0.1192
        float expected = 1.0f / (1.0f + (float)Math.Exp(2f));
        Assert.That(blazepose.landmarks[0].confidence, Is.EqualTo(expected).Within(1e-4f));
    }

    // =======================================================
    // 6. GetResult (affine transform)
    // =======================================================
    [Test]
    public void GetResult_NoScale_ReturnsEmpty()
    {
        // affine_scale is 0 by default -> should return empty
        SetupLandmarksForGetResult();
        blazepose.affine_scale = 0;

        var result = blazepose.GetResult();
        Assert.That(result.Count, Is.EqualTo(0));
    }

    [Test]
    public void GetResult_WithIdentityTransform_19Keypoints()
    {
        SetupLandmarksForGetResult();
        blazepose.affine_scale = 1.0f;
        blazepose.affine_angle = 0;
        blazepose.affine_xc = 0.5f;
        blazepose.affine_yc = 0.5f;

        var result = blazepose.GetResult();
        Assert.That(result.Count, Is.EqualTo(1));
        Assert.That(result[0].points.Length, Is.EqualTo(19));
    }

    [Test]
    public void GetResult_ShoulderCenter_IsAverage()
    {
        SetupLandmarksForGetResult();
        blazepose.affine_scale = 1.0f;
        blazepose.affine_angle = 0;
        blazepose.affine_xc = 0.5f;
        blazepose.affine_yc = 0.5f;

        // Set left shoulder and right shoulder with distinct positions
        blazepose.landmarks[(int)BodyPartIndex.LeftShoulder] = new Landmark
        {
            position = new Vector3(0.3f, 0.4f, 0),
            confidence = 0.9f
        };
        blazepose.landmarks[(int)BodyPartIndex.RightShoulder] = new Landmark
        {
            position = new Vector3(0.7f, 0.6f, 0),
            confidence = 0.8f
        };

        var result = blazepose.GetResult();
        var shoulderCenter = result[0].points[17]; // SHOULDER_CENTER = 17

        // The shoulder center position is average of left and right shoulder, then affine transformed
        // avg position: (0.5, 0.5, 0). Transform: (0.5-0.5)*1+0.5=0.5
        Assert.That(shoulderCenter.x, Is.EqualTo(0.5f).Within(Tolerance), "shoulder center x");
        Assert.That(shoulderCenter.y, Is.EqualTo(0.5f).Within(Tolerance), "shoulder center y");
        // confidence = min(0.9, 0.8) = 0.8
        Assert.That(shoulderCenter.score, Is.EqualTo(0.8f).Within(Tolerance), "shoulder center confidence");
    }

    [Test]
    public void GetResult_BodyCenter_IsAverageOfFourPoints()
    {
        SetupLandmarksForGetResult();
        blazepose.affine_scale = 1.0f;
        blazepose.affine_angle = 0;
        blazepose.affine_xc = 0.5f;
        blazepose.affine_yc = 0.5f;

        blazepose.landmarks[(int)BodyPartIndex.LeftShoulder] = new Landmark
            { position = new Vector3(0.4f, 0.5f, 0), confidence = 0.9f };
        blazepose.landmarks[(int)BodyPartIndex.RightShoulder] = new Landmark
            { position = new Vector3(0.6f, 0.5f, 0), confidence = 0.8f };
        blazepose.landmarks[(int)BodyPartIndex.LeftHip] = new Landmark
            { position = new Vector3(0.4f, 0.5f, 0), confidence = 0.7f };
        blazepose.landmarks[(int)BodyPartIndex.RightHip] = new Landmark
            { position = new Vector3(0.6f, 0.5f, 0), confidence = 0.6f };

        var result = blazepose.GetResult();
        var bodyCenter = result[0].points[18]; // BODY_CENTER = 18

        // avg = (0.5, 0.5) -> affine transform with identity -> (0.5, 0.5)
        Assert.That(bodyCenter.x, Is.EqualTo(0.5f).Within(Tolerance));
        Assert.That(bodyCenter.y, Is.EqualTo(0.5f).Within(Tolerance));
        // confidence = min(0.9, 0.8, 0.7, 0.6) = 0.6
        Assert.That(bodyCenter.score, Is.EqualTo(0.6f).Within(Tolerance));
    }

    [Test]
    public void GetResult_AffineRotation_MatchesPython()
    {
        // Test with 90 degree rotation
        SetupLandmarksForGetResult();
        blazepose.affine_scale = 1.0f;
        blazepose.affine_angle = (float)(Math.PI / 2);  // 90 degrees
        blazepose.affine_xc = 0.5f;
        blazepose.affine_yc = 0.5f;

        // Set nose at (0.75, 0.5) -> after affine: should rotate 90 deg around (0.5, 0.5)
        blazepose.landmarks[(int)BodyPartIndex.Nose] = new Landmark
        {
            position = new Vector3(0.75f, 0.5f, 0),
            confidence = 1.0f
        };

        var result = blazepose.GetResult();
        var nose = result[0].points[0]; // NOSE = 0

        // The affine transform:
        // x = ((px - 0.5) * cos(-angle) + (py - 0.5) * sin(-angle)) * scale + xc
        // y = ((px - 0.5) * -sin(-angle) + (py - 0.5) * cos(-angle)) * scale + yc
        float cs = (float)Math.Cos(-blazepose.affine_angle);
        float ss = (float)Math.Sin(-blazepose.affine_angle);
        float px = 0.75f - 0.5f;
        float py = 0.5f - 0.5f;
        float expectedX = (px * cs + py * ss) * 1.0f + 0.5f;
        float expectedY = (px * -ss + py * cs) * 1.0f + 0.5f;

        Assert.That(nose.x, Is.EqualTo(expectedX).Within(1e-4f), "rotated x");
        Assert.That(nose.y, Is.EqualTo(expectedY).Within(1e-4f), "rotated y");
    }

    // =======================================================
    // 7. BodyPartIndex enum values
    // =======================================================
    [Test]
    public void BodyPartIndex_MatchesPythonMapping()
    {
        Assert.That((int)BodyPartIndex.Nose, Is.EqualTo(0));
        Assert.That((int)BodyPartIndex.LeftShoulder, Is.EqualTo(11));
        Assert.That((int)BodyPartIndex.RightShoulder, Is.EqualTo(12));
        Assert.That((int)BodyPartIndex.LeftHip, Is.EqualTo(23));
        Assert.That((int)BodyPartIndex.RightHip, Is.EqualTo(24));
        Assert.That((int)BodyPartIndex.LeftAnkle, Is.EqualTo(27));
        Assert.That((int)BodyPartIndex.RightAnkle, Is.EqualTo(28));
        Assert.That((int)BodyPartIndex.RightFootIndex, Is.EqualTo(32));
    }

    // =======================================================
    // 8. Constants validation
    // =======================================================
    [Test]
    public void Constants_MatchPythonReference()
    {
        // Detection model input: 224x224x3
        Assert.That(AiliaBlazepose.BLAZEPOSE_DETECTOR_KEYPOINT_COUNT, Is.EqualTo(4));

        // Estimation model: 256x256x3, 33 landmarks * 5 values + 30 auxiliary = 195
        // Output tensor size = 195 (33 landmarks * 5 values each + 30 auxiliary)
    }

    // =======================================================
    // 9. Landmark struct
    // =======================================================
    [Test]
    public void Landmark_StoresPositionAndConfidence()
    {
        var lm = new Landmark
        {
            index = 5,
            position = new Vector3(0.1f, 0.2f, 0.3f),
            confidence = 0.95f
        };

        Assert.That(lm.index, Is.EqualTo(5));
        Assert.That(lm.position.x, Is.EqualTo(0.1f).Within(Tolerance));
        Assert.That(lm.position.y, Is.EqualTo(0.2f).Within(Tolerance));
        Assert.That(lm.position.z, Is.EqualTo(0.3f).Within(Tolerance));
        Assert.That(lm.confidence, Is.EqualTo(0.95f).Within(Tolerance));
    }

    // =======================================================
    // Helpers
    // =======================================================
    private Box CreateBox(float xMin, float yMin, float xMax, float yMax, float score = 1.0f)
    {
        float width = xMax - xMin;
        float height = yMax - yMin;
        return new Box
        {
            xMin = xMin,
            yMin = yMin,
            xMax = xMax,
            yMax = yMax,
            area = width * height,
            score = score,
            keypoints = new Vector2[AiliaBlazepose.BLAZEPOSE_DETECTOR_KEYPOINT_COUNT]
        };
    }

    private void SetupLandmarksForGetResult()
    {
        // Fill with 33 landmarks at center with high confidence
        blazepose.landmarks = new List<Landmark>();
        for (int i = 0; i < 33; i++)
        {
            blazepose.landmarks.Add(new Landmark
            {
                position = new Vector3(0.5f, 0.5f, 0),
                confidence = 1.0f
            });
        }
    }
}
