using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

using ailiaSDK;
using ailia;

struct Box
{
    public float xMin;
    public float xMax;
    public float yMin;
    public float yMax;
    public float area;
    public float score;
    public Vector2[] keypoints;

    private int mergedBoxCount;

    public float GetJaccardOverlap(Box other)
    {
        float widthOverlap = Math.Max(0, Math.Min(xMax, other.xMax) - Math.Max(xMin, other.xMin));
        float heightOverlap = Math.Max(0, Math.Min(yMax, other.yMax) - Math.Max(yMin, other.yMin));

        float intersection = widthOverlap * heightOverlap;

        float union = area + other.area - intersection;

        return intersection / union;
    }

    public void Merge(Box other)
    {
        if (mergedBoxCount == 0)
        {
            xMin *= score;
            yMin *= score;
            xMax *= score;
            yMax *= score;

            for (int i = 0; i < AiliaBlazepose.BLAZEPOSE_DETECTOR_KEYPOINT_COUNT; ++i)
            {
                keypoints[i] *= score; 
            }
        }

        xMin += other.score * other.xMin;
        yMin += other.score * other.yMin;
        xMax += other.score * other.xMax;
        yMax += other.score * other.yMax;
        
        for (int i = 0; i < AiliaBlazepose.BLAZEPOSE_DETECTOR_KEYPOINT_COUNT; ++i)
        {
            keypoints[i] += other.score * other.keypoints[i];
        }
        
        score += other.score;
        mergedBoxCount += Math.Max(1, other.mergedBoxCount);
    }

    public void FinalizeMerge()
    {
        if (mergedBoxCount == 0)
        {
            return;
        }

        xMin /= score;
        yMin /= score;
        xMax /= score;
        yMax /= score;

        for (int i = 0; i < AiliaBlazepose.BLAZEPOSE_DETECTOR_KEYPOINT_COUNT; ++i)
        {
            keypoints[i] /= score;
        }

        score /= (1 + mergedBoxCount);

        mergedBoxCount = 0;
    }
}

public struct Landmark
{
    public int index;
    public Vector3 position;
    public float confidence;
}

public struct Transformation
{
    public Vector3 translation;
    public float angle;
    public Vector3 scale;
}

public enum BodyPartIndex
{
    Nose = 0,
    InnerLeftEye,
    LeftEye,
    OuterLeftEye,
    InnerRightEye,
    RightEye,
    OuterRightEye,
    LeftEar,
    RightEar,
    LeftMouth,
    RightMouth,

    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftPinky,
    RightPinky,
    LeftIndex,
    RightIndex,
    LeftThumb,
    RightThumb,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
    LeftHeel,
    RightHeel,
    LeftFootIndex,
    RightFootIndex,
}

public class AiliaBlazepose : IDisposable
{
#if UNITY_2017_1_OR_NEWER
    public ComputeShader computeShader = null;
#endif

    private AiliaModel ailiaPoseDetection = new AiliaModel();
    private AiliaModel ailiaPoseEstimation = new AiliaModel();

    private static readonly int BLAZEPOSE_DETECTOR_INPUT_RESOLUTION = 224;
    private static readonly uint BLAZEPOSE_DETECTOR_INPUT_CHANNEL_COUNT = 3;
    private static readonly uint BLAZEPOSE_DETECTOR_TENSOR_COUNT = 2254;
    private static readonly uint BLAZEPOSE_DETECTOR_TENSOR_SIZE = 12;
    public static readonly uint BLAZEPOSE_DETECTOR_KEYPOINT_COUNT = 4;
    private static readonly float BLAZEPOSE_DETECTOR_RAW_SCORE_THRESHOLD = 100;
    private static readonly float BLAZEPOSE_DETECTOR_MINIMUM_SCORE_THRESHOLD = 0.5f;
    private static readonly float BLAZEPOSE_DETECTOR_MINIMUM_OVERLAP_THRESHOLD = 0.3f;
    internal float[,] anchors;

    private static readonly int BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION = 256;
    private static readonly int BLAZEPOSE_ESTIMATOR_TENSOR_COUNT = 1;
    private static readonly int BLAZEPOSE_ESTIMATOR_TENSOR_SIZE = 195;

#if UNITY_2017_1_OR_NEWER
    int kernelIndex = -1;
    int ID_InputTexture = Shader.PropertyToID("InputTexture");
    int ID_InputWidth = Shader.PropertyToID("InputWidth");
    int ID_InputHeight = Shader.PropertyToID("InputHeight");
    int ID_OutputTexture = Shader.PropertyToID("OutputTexture");
    int ID_OutputWidth = Shader.PropertyToID("OutputWidth");
    int ID_OutputHeight = Shader.PropertyToID("OutputHeight");
    int ID_Matrix = Shader.PropertyToID("Matrix");
    int ID_BackgroundColor = Shader.PropertyToID("BackgroundColor");
#endif

    struct JsonFloatArray
    {
        public float[] array;
    }

    private bool EnvironmentWithoutFP16(AiliaModel model, int type)
    {
        int count = model.GetEnvironmentCount();
        if (count == -1)
        {
            return false;
        }

        for (int i = 0; i < count; i++)
        {
            Ailia.AILIAEnvironment env = model.GetEnvironment(i);
            if (env == null)
            {
                return false;
            }

            if (env.type == type)
            {
                if ((env.props & Ailia.AILIA_ENVIRONMENT_PROPERTY_FP16) != 0)
                {
                    continue;
                }
                if (!model.SelectEnvironment(i))
                {
                    return false;
                }
                if (env.backend == Ailia.AILIA_ENVIRONMENT_BACKEND_CUDA || env.backend == Ailia.AILIA_ENVIRONMENT_BACKEND_VULKAN)
                {
                    return true;    //優先
                }
            }
        }
        return true;
    }

    public AiliaBlazepose(bool gpuMode, string assetPath, string jsonPath)
    {
        bool status;

        if (gpuMode)
        {
            EnvironmentWithoutFP16(ailiaPoseDetection, Ailia.AILIA_ENVIRONMENT_TYPE_GPU);
        }
        string modelName = "pose_detection";
        status = ailiaPoseDetection.OpenFile($"{assetPath}/{modelName}.onnx.prototxt", $"{assetPath}/{modelName}.onnx");
        if (status == false)
        {
            string message = $"Could not load model {modelName}";
            Debug.LogError(message);
            throw new Exception(message);
        }
        Debug.Log($"Model loaded {modelName}");

        if (gpuMode)
        {
            EnvironmentWithoutFP16(ailiaPoseEstimation, Ailia.AILIA_ENVIRONMENT_TYPE_GPU);
        }
        modelName = "pose_landmark_heavy";
        status = ailiaPoseEstimation.OpenFile($"{assetPath}/{modelName}.onnx.prototxt", $"{assetPath}/{modelName}.onnx");
        if (status == false)
        {
            string message = $"Could not load model {modelName}";
            Debug.LogError(message);
            throw new Exception(message);
        }

        Debug.Log($"Model loaded {modelName}");

        string anchorsJSON = File.ReadAllText($"{jsonPath}/blazepose_anchors.json");
        float[] anchorsFlat = JsonUtility.FromJson<JsonFloatArray>($"{{ \"array\": {anchorsJSON} }}").array;

        anchors = new float[BLAZEPOSE_DETECTOR_TENSOR_COUNT, 4];

        for (int i = 0; i < anchorsFlat.Length; ++i)
        {
            anchors[i / 4, i % 4] = anchorsFlat[i];
        }

        inputArray = new float[BLAZEPOSE_DETECTOR_INPUT_RESOLUTION * BLAZEPOSE_DETECTOR_INPUT_RESOLUTION * BLAZEPOSE_DETECTOR_INPUT_CHANNEL_COUNT];
        rawBoxesOutput = new float[BLAZEPOSE_DETECTOR_TENSOR_COUNT * BLAZEPOSE_DETECTOR_TENSOR_SIZE];
        rawScoresOutput = new float[BLAZEPOSE_DETECTOR_TENSOR_COUNT];
        estimationScoreBuffer = new float[1];
        estimationInputArray = new float[BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION * BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION * BLAZEPOSE_DETECTOR_INPUT_CHANNEL_COUNT];
        estimationOutputBuffer = new float[BLAZEPOSE_ESTIMATOR_TENSOR_COUNT * BLAZEPOSE_ESTIMATOR_TENSOR_SIZE];
    }

    // Internal constructor for unit testing (no model loading)
    internal AiliaBlazepose()
    {
        inputArray = new float[BLAZEPOSE_DETECTOR_INPUT_RESOLUTION * BLAZEPOSE_DETECTOR_INPUT_RESOLUTION * BLAZEPOSE_DETECTOR_INPUT_CHANNEL_COUNT];
        rawBoxesOutput = new float[BLAZEPOSE_DETECTOR_TENSOR_COUNT * BLAZEPOSE_DETECTOR_TENSOR_SIZE];
        rawScoresOutput = new float[BLAZEPOSE_DETECTOR_TENSOR_COUNT];
        estimationScoreBuffer = new float[1];
        estimationInputArray = new float[BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION * BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION * BLAZEPOSE_DETECTOR_INPUT_CHANNEL_COUNT];
        estimationOutputBuffer = new float[BLAZEPOSE_ESTIMATOR_TENSOR_COUNT * BLAZEPOSE_ESTIMATOR_TENSOR_SIZE];
        anchors = new float[BLAZEPOSE_DETECTOR_TENSOR_COUNT, 4];
    }

    internal float[] inputArray;
    internal float[] rawBoxesOutput;
    internal float[] rawScoresOutput;
    internal List<Box> boxes;
    private Box? poseDetectionBox;
#if UNITY_2017_1_OR_NEWER
    RenderTexture preprocessBuffer;
    private void PreprocessTexture(Texture2D texture)
    {
        if(preprocessBuffer == null)
		{
            preprocessBuffer = new RenderTexture(BLAZEPOSE_DETECTOR_INPUT_RESOLUTION, BLAZEPOSE_DETECTOR_INPUT_RESOLUTION, 0, RenderTextureFormat.ARGB32);
        }

        Texture2D letterboxed = TexturePreprocessor.PreprocessTexture(texture, preprocessBuffer, BLAZEPOSE_DETECTOR_INPUT_RESOLUTION * Vector2.one);
		Color32[] colorData = letterboxed.GetPixels32();
        const float factor = 1 / 255f;

        for (int heightIndex = 0; heightIndex < BLAZEPOSE_DETECTOR_INPUT_RESOLUTION; heightIndex++)
        {
            for (int widthIndex = 0; widthIndex < BLAZEPOSE_DETECTOR_INPUT_RESOLUTION; widthIndex++)
            {
                int index = (int) (((heightIndex * BLAZEPOSE_DETECTOR_INPUT_RESOLUTION) + widthIndex) * BLAZEPOSE_DETECTOR_INPUT_CHANNEL_COUNT);
				Color32 value = colorData[(BLAZEPOSE_DETECTOR_INPUT_RESOLUTION - heightIndex - 1) * BLAZEPOSE_DETECTOR_INPUT_RESOLUTION + widthIndex];

				inputArray[index + 0] = value.r * factor;
				inputArray[index + 1] = value.g * factor;
				inputArray[index + 2] = value.b * factor;
			}
        }
    }
#endif

    internal uint estimationInputWidth;
    internal uint estimationInputHeight;
    internal float[] estimationInputArray;
    internal float[] estimationOutputBuffer;
    internal float[] estimationScoreBuffer;
    public List<Landmark> landmarks = new List<Landmark>();

#if UNITY_2017_1_OR_NEWER
    private void PreprocessTextureEstimation(Texture2D texture)
    {
        estimationInputWidth = ((uint)texture.width);
        estimationInputHeight = ((uint)texture.height);

        Color32[] colorData = texture.GetPixels32();
        const float factor = 1 / 255f;

        for (int heightIndex = 0; heightIndex < estimationInputHeight; heightIndex++)
        {
            for (int widthIndex = 0; widthIndex < estimationInputWidth; widthIndex++)
            {
                int index = (int)(((heightIndex * estimationInputWidth) + widthIndex) * BLAZEPOSE_DETECTOR_INPUT_CHANNEL_COUNT);
				Color32 value = colorData[heightIndex * estimationInputWidth + widthIndex];

				estimationInputArray[index + 0] = value.r * factor;
                estimationInputArray[index + 1] = value.g * factor;
                estimationInputArray[index + 2] = value.b * factor;
            }
        }
    }
#endif

    internal float affine_xc=0;
    internal float affine_yc=0;
    internal float affine_x1=0;
    internal float affine_y1=0;
    internal float affine_scale=0;
    internal float affine_angle=0;

#if UNITY_2017_1_OR_NEWER
    private Texture2D ExtractROIFromBox(Texture2D texture, Box box)
    {
        float finalSquareLength = Mathf.Max(texture.width, texture.height);
        int xOffset = ((int) ((finalSquareLength - texture.width) / 2));
        int yOffset = ((int) ((finalSquareLength - texture.height) / 2));

        Box scaledBox = box;
		scaledBox.xMin = finalSquareLength * scaledBox.xMin - xOffset;
		scaledBox.xMax = finalSquareLength * scaledBox.xMax - xOffset;
		scaledBox.yMin = texture.height - 1 - finalSquareLength * scaledBox.yMin + yOffset; 
        scaledBox.yMax = texture.height - 1 - finalSquareLength * scaledBox.yMax + yOffset;
        scaledBox.keypoints = new Vector2[scaledBox.keypoints.Length];

        for (int i = 0; i < box.keypoints.Length; ++i)
        {
            scaledBox.keypoints[i] = new Vector2(
				finalSquareLength * box.keypoints[i].x - xOffset,
				texture.height - 1 - finalSquareLength * box.keypoints[i].y + yOffset
            );
        }

        int kp1 = 0;
        int kp2 = 1;
        float theta0 = Mathf.PI / 2f;
        float dscale = 1.1f;

        float xc = scaledBox.keypoints[kp1].x;
        float yc = scaledBox.keypoints[kp1].y;
        float x1 = scaledBox.keypoints[kp2].x;
        float y1 = scaledBox.keypoints[kp2].y;
        float scale = dscale * Mathf.Sqrt((Mathf.Pow(xc - x1, 2) + Mathf.Pow(yc - y1, 2))) * 2;
        float angle = Mathf.Atan2(yc - y1, xc - x1) - theta0;

        affine_xc = box.keypoints[kp1].x;
        affine_yc = box.keypoints[kp1].y;
        affine_x1 = box.keypoints[kp2].x;
        affine_y1 = box.keypoints[kp2].y;
        affine_scale = dscale * Mathf.Sqrt((Mathf.Pow(affine_xc - affine_x1, 2) + Mathf.Pow(affine_yc - affine_y1, 2))) * 2;
        affine_angle = Mathf.Atan2(affine_yc - affine_y1, affine_xc - affine_x1) - theta0;
        
        Vector2[] points = new Vector2[]
        {
            new Vector2(1, -1),
            new Vector2(1, 1),
            new Vector2(-1, -1)
        };

        for (int i = 0; i < points.Length; ++i) {
            points[i] *= scale / 2;
        }

        float cosAngle = Mathf.Cos(angle);
        float sinAngle = Mathf.Sin(angle);

        for (int i = 0; i < points.Length; ++i)
        {
            Vector2 p = points[i];
            points[i] = new Vector2(
                p.x * cosAngle - p.y * sinAngle + xc,
                p.x * sinAngle + p.y * cosAngle + yc
            );
        }

        int resolution = BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION - 1;
        Matrix4x4 before_m = new Matrix4x4() {
            m00 = 0, m01 = 0, m02 = resolution, m03 = 0,
            m10 = 0, m11 = resolution, m12 = 0, m13 = 0,
            m20 = 1, m21 = 1, m22 = 1, m23 = 0,
            m30 = 0, m31 = 0, m32 = 0, m33 = 1,
        };

        Matrix4x4 after_m = new Matrix4x4()
        {
            m00 = points[0].x, m01 = points[1].x, m02 = points[2].x, m03 = 0,
            m10 = points[0].y, m11 = points[1].y, m12 = points[2].y, m13 = 0,
            m20 = 1, m21 = 1, m22 = 1, m23 = 0,
            m30 = 0, m31 = 0, m32 = 0, m33 = 1,
        };

        Matrix4x4 transfrom_m = after_m * before_m.inverse;



        if(computeTexture == null)
		{
            computeTexture = new RenderTexture(BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION, BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION, 32);
            computeTexture.enableRandomWrite = true;
        }

        if(kernelIndex < 0)
		{
            kernelIndex = computeShader.FindKernel("AffineTransform");
		}

        computeShader.SetTexture(kernelIndex, ID_InputTexture, texture);
        computeShader.SetTexture(kernelIndex, ID_OutputTexture, computeTexture);
        computeShader.SetMatrix(ID_Matrix, transfrom_m);
        computeShader.SetInt(ID_InputWidth, texture.width);
        computeShader.SetInt(ID_InputHeight, texture.height);
        computeShader.SetInt(ID_OutputWidth, BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION);
        computeShader.SetInt(ID_OutputHeight, BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION);
        computeShader.SetVector(ID_BackgroundColor, new Vector4(0, 0, 0, 1));

        computeShader.Dispatch(kernelIndex, BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION / 32 + 1, BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION / 32 + 1, 1);

        roiTexture = toTexture2D(computeTexture, roiTexture);

        return roiTexture;
    }
    RenderTexture computeTexture;
    Texture2D roiTexture;

    Texture2D toTexture2D(RenderTexture rTex, Texture2D output = null)
    {
        var org = RenderTexture.active;
		output = output ?? new Texture2D(rTex.width, rTex.height, TextureFormat.RGBA32, false);
        RenderTexture.active = rTex;
        output.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        output.Apply();
        RenderTexture.active = org;
        return output;
    }
#endif

#if UNITY_2017_1_OR_NEWER
     Texture2D input_texture=null;

    public List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose> RunPoseEstimation(Color32 [] camera, int tex_width, int tex_height)
    {
        if(input_texture==null){
            input_texture = new Texture2D(tex_width, tex_height);
        }
		input_texture.SetPixels32(camera);
		input_texture.Apply();

        Texture2D detection = RunDetectionModel(input_texture);
        if (detection == null)
        {
            Debug.Log("NO POSE");
			return new List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose>();
        }

		RunEstimationModel(detection);

        return GetResult();
    }
#endif

#if UNITY_2017_1_OR_NEWER
    private Texture2D RunDetectionModel(Texture2D inputTexture)
    {
        bool status;

		if (poseDetectionBox == null)
		{
            PreprocessTexture(inputTexture);

            int inputBlobIndex = ailiaPoseDetection.FindBlobIndexByName("input_1");

            status = ailiaPoseDetection.SetInputBlobShape(
                new Ailia.AILIAShape
                {
                    x = (uint)BLAZEPOSE_DETECTOR_INPUT_CHANNEL_COUNT,
                    y = (uint)BLAZEPOSE_DETECTOR_INPUT_RESOLUTION,
                    z = (uint)BLAZEPOSE_DETECTOR_INPUT_RESOLUTION,
                    w = 1,
                    dim = 4
                },
                inputBlobIndex
            );

            if (status == false)
            {
                Debug.LogError("Could not set input blob shape");
                Debug.LogError(ailiaPoseDetection.GetErrorDetail());
            }

            status = ailiaPoseDetection.SetInputBlobData(inputArray, inputBlobIndex);
            if (status == false)
            {
                Debug.LogError("Could not set input blob data");
                Debug.LogError(ailiaPoseDetection.GetErrorDetail());
            }

            bool result = ailiaPoseDetection.Update();
            if (result == false)
            {
                Debug.Log(ailiaPoseDetection.GetErrorDetail());
            }

            int outputBlobIndex = ailiaPoseDetection.FindBlobIndexByName("Identity");
            status = ailiaPoseDetection.GetBlobData(rawBoxesOutput, outputBlobIndex);
            if (status == false)
            {
                Debug.LogError("Could not get output blob data " + outputBlobIndex);
                Debug.LogError(ailiaPoseDetection.GetErrorDetail());
            }

            outputBlobIndex = ailiaPoseDetection.FindBlobIndexByName("Identity_1");
            status = ailiaPoseDetection.GetBlobData(rawScoresOutput, outputBlobIndex);
            if (status == false)
            {
                Debug.LogError("Could not get output blob data " + outputBlobIndex);
                Debug.LogError(ailiaPoseDetection.GetErrorDetail());
            }

            DecodeAndProcessBoxes();

            if (boxes.Count == 0)
            {
				return null;
			}
            else
            {
                poseDetectionBox = boxes[0];
            }
        }

        Texture2D roi = ExtractROIFromBox(inputTexture, poseDetectionBox.Value);
		return roi;
    }

    private void RunEstimationModel(Texture2D inputTexture)
    {

		bool status;
		PreprocessTextureEstimation(inputTexture);

        int inputBlobIndex = ailiaPoseEstimation.FindBlobIndexByName("input_1");

        status = ailiaPoseEstimation.SetInputBlobShape(
            new Ailia.AILIAShape
            {
                x = (uint)BLAZEPOSE_DETECTOR_INPUT_CHANNEL_COUNT,
                y = (uint)BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION,
                z = (uint)BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION,
                w = 1,
                dim = 4
            },
            inputBlobIndex
        );

        if (status == false)
        {
            Debug.LogError("Could not set input blob shape");
            Debug.LogError(ailiaPoseEstimation.GetErrorDetail());
        }

        status = ailiaPoseEstimation.SetInputBlobData(estimationInputArray, inputBlobIndex);
        if (status == false)
        {
            Debug.LogError("Could not set input blob data");
            Debug.LogError(ailiaPoseEstimation.GetErrorDetail());
        }

        status = ailiaPoseEstimation.Update();
        if (status == false)
        {
            Debug.LogError(ailiaPoseEstimation.GetErrorDetail());
            return;
        }

        int outputBlobIndex = ailiaPoseEstimation.FindBlobIndexByName("Identity_1");
        status = ailiaPoseEstimation.GetBlobData(estimationScoreBuffer, outputBlobIndex);
        if (status == false)
        {
            Debug.LogError("Could not get pose score output blob data " + outputBlobIndex);
            Debug.LogError(ailiaPoseEstimation.GetErrorDetail());
        }
        float poseScore = estimationScoreBuffer[0];

        outputBlobIndex = ailiaPoseEstimation.FindBlobIndexByName("Identity");
        status = ailiaPoseEstimation.GetBlobData(estimationOutputBuffer, outputBlobIndex);

        if (status == false)
        {
            Debug.LogError("Could not get output blob data " + outputBlobIndex);
            Debug.LogError(ailiaPoseEstimation.GetErrorDetail());
        }

        DecodeAndProcessLandmarks();
    }
#endif

    internal float Sigmoid(float x)
    {
        return 1.0f / (1.0f + Mathf.Exp(-x));
    }

    internal void DecodeAndProcessBoxes()
    {
        List<Box> remainingBoxes = new List<Box>();
        boxes = new List<Box>();

        float xScale = BLAZEPOSE_DETECTOR_INPUT_RESOLUTION;
        float yScale = BLAZEPOSE_DETECTOR_INPUT_RESOLUTION;
        float wScale = BLAZEPOSE_DETECTOR_INPUT_RESOLUTION;
        float hScale = BLAZEPOSE_DETECTOR_INPUT_RESOLUTION;

        Func<int, int, float> getFloat = (tI, cI) => rawBoxesOutput[tI * BLAZEPOSE_DETECTOR_TENSOR_SIZE + cI];

        for (int tI = 0; tI < BLAZEPOSE_DETECTOR_TENSOR_COUNT; ++tI)
        {
            float score = Sigmoid(Mathf.Clamp(rawScoresOutput[tI], -BLAZEPOSE_DETECTOR_RAW_SCORE_THRESHOLD, BLAZEPOSE_DETECTOR_RAW_SCORE_THRESHOLD));

            if (score < BLAZEPOSE_DETECTOR_MINIMUM_SCORE_THRESHOLD)
            {
                continue;
            }

            float xCenter = getFloat(tI, 0) / xScale * anchors[tI, 2] + anchors[tI, 0];
            float yCenter = getFloat(tI, 1) / yScale * anchors[tI, 3] + anchors[tI, 1];
            float width = getFloat(tI, 2) / wScale * anchors[tI, 2];
            float height = getFloat(tI, 3) / hScale * anchors[tI, 3];

            Vector2[] keypoints = new Vector2[BLAZEPOSE_DETECTOR_KEYPOINT_COUNT];

            for (int i = 0; i < BLAZEPOSE_DETECTOR_KEYPOINT_COUNT; ++i)
            {
                int index = 4 + 2 * i;
                keypoints[i] = new Vector2(
                    getFloat(tI, index) / xScale * anchors[tI, 2] + anchors[tI, 0],
                    getFloat(tI, index + 1) / yScale * anchors[tI, 3] + anchors[tI, 1]
                );
            }

            remainingBoxes.Add(new Box
            {
                xMin = xCenter - width / 2,
                yMin = yCenter - height / 2,
                xMax = xCenter + width / 2,
                yMax = yCenter + height / 2,
                keypoints = keypoints,
                area = width * height,
                score = score
            });
        }

        // Merge boxes by weighted non-max suppression
        remainingBoxes.Sort((a, b) => Math.Sign(b.score - a.score));

        while (remainingBoxes.Count > 0)
        {
            Box referenceBox = remainingBoxes[0];
            Box mergedBox = referenceBox;
            remainingBoxes.RemoveAt(0);

            for (int i = 0; i < remainingBoxes.Count; ++i)
            {
                if (referenceBox.GetJaccardOverlap(remainingBoxes[i]) > BLAZEPOSE_DETECTOR_MINIMUM_OVERLAP_THRESHOLD)
                {
                    mergedBox.Merge(remainingBoxes[i]);
                    remainingBoxes.RemoveAt(i);
                    --i;
                }
            }

            mergedBox.FinalizeMerge();
            boxes.Add(mergedBox);
        }
    }

    internal void DecodeAndProcessLandmarks()
    {
        landmarks = new List<Landmark>();

        for (int i = 0; i < 33; ++i)
        {
            float x = estimationOutputBuffer[i * 5] / BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION;
            float y = estimationOutputBuffer[i * 5 + 1] / BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION;
            float z = estimationOutputBuffer[i * 5 + 2] / BLAZEPOSE_ESTIMATOR_INPUT_RESOLUTION;
            float visibility = estimationOutputBuffer[i * 5 + 3];
            float presence = estimationOutputBuffer[i * 5 + 4];

            landmarks.Add(new Landmark
            {
                position = new Vector3(x, y, z),
                confidence = Sigmoid(Math.Min(visibility, presence))
            }); ;
        }
    }

    public List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose> GetResult(){
        List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose> result_list=new List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose>();
        int [] keypoint_list={
            (int)BodyPartIndex.Nose,
            (int)BodyPartIndex.LeftEye,
            (int)BodyPartIndex.RightEye,
            (int)BodyPartIndex.LeftEar,
            (int)BodyPartIndex.RightEar,
            (int)BodyPartIndex.LeftShoulder,
            (int)BodyPartIndex.RightShoulder,
            (int)BodyPartIndex.LeftElbow,
            (int)BodyPartIndex.RightElbow,
            (int)BodyPartIndex.LeftWrist,
            (int)BodyPartIndex.RightWrist,
            (int)BodyPartIndex.LeftHip,
            (int)BodyPartIndex.RightHip,
            (int)BodyPartIndex.LeftKnee,
            (int)BodyPartIndex.RightKnee,
            (int)BodyPartIndex.LeftAnkle,
            (int)BodyPartIndex.RightAnkle};

        if(affine_scale==0){
            return result_list;
        }

        float cs=(float)Math.Cos(-affine_angle);
        float ss=(float)Math.Sin(-affine_angle);

        AiliaPoseEstimator.AILIAPoseEstimatorObjectPose one_pose=new AiliaPoseEstimator.AILIAPoseEstimatorObjectPose();
        one_pose.points = new AiliaPoseEstimator.AILIAPoseEstimatorKeypoint[19];
        for(int i=0;i<AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_CNT;i++){
            Vector3 pos = Vector3.zero;
            float conf = 0;
            if(i<=AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_ANKLE_RIGHT){
                pos = landmarks[keypoint_list[i]].position;
                conf = landmarks[keypoint_list[i]].confidence;
            }
            if(i==AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_SHOULDER_CENTER){
                pos = (landmarks[(int)BodyPartIndex.LeftShoulder].position + landmarks[(int)BodyPartIndex.RightShoulder].position)/2;
                conf = Math.Min(landmarks[(int)BodyPartIndex.LeftShoulder].confidence,landmarks[(int)BodyPartIndex.RightShoulder].confidence);
            }
            if(i==AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_BODY_CENTER){
                pos = (landmarks[(int)BodyPartIndex.LeftHip].position + landmarks[(int)BodyPartIndex.RightHip].position + landmarks[(int)BodyPartIndex.LeftShoulder].position + landmarks[(int)BodyPartIndex.RightShoulder].position)/4;
                conf = Math.Min(Math.Min(landmarks[(int)BodyPartIndex.LeftHip].confidence, landmarks[(int)BodyPartIndex.RightHip].confidence),Math.Min(landmarks[(int)BodyPartIndex.LeftShoulder].confidence, landmarks[(int)BodyPartIndex.RightShoulder].confidence));
            }
            AiliaPoseEstimator.AILIAPoseEstimatorKeypoint keypoint =new AiliaPoseEstimator.AILIAPoseEstimatorKeypoint();
            keypoint.x = ((pos.x - 0.5f) *  cs + (pos.y - 0.5f) * ss) * affine_scale + affine_xc;
            keypoint.y = ((pos.x - 0.5f) * -ss + (pos.y - 0.5f) * cs) * affine_scale + affine_yc;
            keypoint.z_local = pos.z;
            keypoint.score = conf;
            one_pose.points[i] = keypoint;
        }
        result_list.Add(one_pose);
        return result_list;
    }

    #region IDisposable Support
    private bool disposedValue = false; // To detect redundant calls

	protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            ailiaPoseDetection.Close();
            ailiaPoseEstimation.Close();
            ailiaPoseDetection = null;
            ailiaPoseEstimation = null;
#if UNITY_2017_1_OR_NEWER
            computeTexture?.Release();
            preprocessBuffer?.Release();
#endif

            disposedValue = true;
        }
    }

    // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
    ~AiliaBlazepose()
    {
      // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
      Dispose(false);
    }

    // This code added to correctly implement the disposable pattern.
    public void Dispose()
    {
        // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
        Dispose(true);
        // TODO: uncomment the following line if the finalizer is overridden above.
        GC.SuppressFinalize(this);
    }
    #endregion

    public string EnvironmentName(){
        return ailiaPoseDetection.EnvironmentName();
    }
}
