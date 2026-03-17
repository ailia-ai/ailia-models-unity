/* BlazePose Fullbody Inference Test using ailia SDK */
/*
 * End-to-end inference test using the real ailia SDK.
 * Downloads models and runs the same blob API pipeline as Unity:
 *   FindBlobIndexByName -> SetInputBlobShape -> SetInputBlobData -> Update -> GetBlobData
 *
 * This test reproduces the inference path in AiliaBlazepose.RunEstimationModel
 * to verify that SetInputBlobShape + Update works correctly with the estimation model.
 *
 * Models:
 *   - pose_detection.onnx (224x224x3 NHWC)
 *   - pose_landmark_heavy.onnx (256x256x3 NHWC)
 */

using NUnit.Framework;
using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text.RegularExpressions;
using UnityEngine;
using ailia;

[TestFixture]
public class BlazePoseInferenceTest
{
    private const string MODEL_DIR = "/tmp/blazepose_models";
    private const string MODEL_BASE_URL = "https://storage.googleapis.com/ailia-models/blazepose-fullbody";

    private static readonly string[] DETECTION_FILES = {
        "pose_detection.onnx",
        "pose_detection.onnx.prototxt"
    };

    private static readonly string[] ESTIMATION_FILES = {
        "pose_landmark_heavy.onnx",
        "pose_landmark_heavy.onnx.prototxt"
    };

    [SetUp]
    public void SetUp()
    {
        Directory.CreateDirectory(MODEL_DIR);
        CheckAndDownloadLicense();
    }

    private static void CheckAndDownloadLicense()
    {
        string homePath = Environment.GetFolderPath(Environment.SpecialFolder.Personal);
        if (string.IsNullOrEmpty(homePath))
            homePath = Environment.GetEnvironmentVariable("HOME") ?? "";
        string licFolder;
        if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows))
        {
            licFolder = Environment.CurrentDirectory;
        }
        else if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                     System.Runtime.InteropServices.OSPlatform.OSX))
        {
            licFolder = Path.Combine(homePath, "Library/SHALO/");
        }
        else
        {
            licFolder = Path.Combine(homePath, ".shalo/");
        }

        string licFile = Path.Combine(licFolder, "AILIA.lic");
        if (IsLicenseValid(licFile))
            return;

        Console.WriteLine("Downloading license file for ailia SDK...");
        Directory.CreateDirectory(licFolder);
        try
        {
            using var httpClient = new HttpClient();
            httpClient.BaseAddress = new Uri("https://axip-console.appspot.com");
            HttpResponseMessage response = httpClient.GetAsync("/license/download/product/AILIA").Result;
            if (response.IsSuccessStatusCode)
            {
                byte[] licenseFile = response.Content.ReadAsByteArrayAsync().Result;
                File.WriteAllBytes(licFile, licenseFile);
                Console.WriteLine($"License saved to {licFile}");
            }
            else
            {
                Console.WriteLine($"License download failed: HTTP {response.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"License download failed: {ex.Message}");
        }
    }

    private static bool IsLicenseValid(string licPath)
    {
        if (!File.Exists(licPath))
            return false;

        string content = File.ReadAllText(licPath).Replace("\r\n", "\n");
        string header = "--- shalo license file ---\naxell:ailia\n";
        if (!content.StartsWith(header))
            return false;

        string[] lines = content.Split('\n');
        Match match = Regex.Match(lines[2], @"(\d{4})/(\d{2})/(\d{2})");
        if (!match.Success)
            return false;

        DateTime expiryDate = new DateTime(
            int.Parse(match.Groups[1].Value),
            int.Parse(match.Groups[2].Value),
            int.Parse(match.Groups[3].Value), 23, 59, 59);
        return DateTime.Now <= expiryDate;
    }

    private void DownloadIfMissing(string filename)
    {
        string path = Path.Combine(MODEL_DIR, filename);
        if (!File.Exists(path))
        {
            Console.WriteLine($"Downloading {filename}...");
            using var client = new WebClient();
            client.DownloadFile($"{MODEL_BASE_URL}/{filename}", path);
        }
    }

    private bool TryDownloadModels(string[] files)
    {
        try
        {
            foreach (var f in files)
                DownloadIfMissing(f);
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Model download failed: {ex.Message}");
            return false;
        }
    }

    private string ModelPath(string name) => Path.Combine(MODEL_DIR, name);

    // =======================================================
    // 1. Detection model inference (blob API)
    // =======================================================
    [Test]
    public void DetectionModel_BlobApi_Inference()
    {
        if (!TryDownloadModels(DETECTION_FILES))
            Assert.Ignore("Could not download detection model");

        AiliaModel model = new AiliaModel();
        try
        {
            bool status = model.OpenFile(
                ModelPath("pose_detection.onnx.prototxt"),
                ModelPath("pose_detection.onnx"));
            if (!status)
            {
                Console.WriteLine($"OpenFile failed: {model.GetErrorDetail()}");
                Assert.Ignore("Could not open detection model (ailia native library may not be available)");
            }

            // Input: NHWC [1, 224, 224, 3] -> AILIAShape: x=3, y=224, z=224, w=1
            int inputIdx = model.FindBlobIndexByName("input_1");
            Assert.That(inputIdx, Is.GreaterThanOrEqualTo(0), "input_1 blob found");

            status = model.SetInputBlobShape(new Ailia.AILIAShape
            {
                x = 3, y = 224, z = 224, w = 1, dim = 4
            }, inputIdx);
            Assert.That(status, Is.True, $"SetInputBlobShape: {model.GetErrorDetail()}");

            float[] input = new float[1 * 224 * 224 * 3];
            // Fill with gray (0.5)
            for (int i = 0; i < input.Length; i++)
                input[i] = 0.5f;

            status = model.SetInputBlobData(input, inputIdx);
            Assert.That(status, Is.True, $"SetInputBlobData: {model.GetErrorDetail()}");

            status = model.Update();
            Assert.That(status, Is.True, $"Update failed: {model.GetErrorDetail()}");

            // Get outputs
            int boxIdx = model.FindBlobIndexByName("Identity");
            int scoreIdx = model.FindBlobIndexByName("Identity_1");
            Assert.That(boxIdx, Is.GreaterThanOrEqualTo(0), "Identity blob found");
            Assert.That(scoreIdx, Is.GreaterThanOrEqualTo(0), "Identity_1 blob found");

            float[] boxes = new float[2254 * 12];
            float[] scores = new float[2254];
            status = model.GetBlobData(boxes, boxIdx);
            Assert.That(status, Is.True, $"GetBlobData boxes: {model.GetErrorDetail()}");
            status = model.GetBlobData(scores, scoreIdx);
            Assert.That(status, Is.True, $"GetBlobData scores: {model.GetErrorDetail()}");

            Console.WriteLine($"Detection model inference succeeded");
            Console.WriteLine($"Boxes range: [{boxes.Min():F4}, {boxes.Max():F4}]");
            Console.WriteLine($"Scores range: [{scores.Min():F4}, {scores.Max():F4}]");
        }
        catch (DllNotFoundException ex)
        {
            Assert.Ignore($"ailia native library not found: {ex.Message}");
        }
        finally
        {
            model.Close();
        }
    }

    // =======================================================
    // 2. Estimation model inference (blob API) - reproduces the error path
    // =======================================================
    [Test]
    public void EstimationModel_BlobApi_Inference()
    {
        if (!TryDownloadModels(ESTIMATION_FILES))
            Assert.Ignore("Could not download estimation model");

        AiliaModel model = new AiliaModel();
        try
        {
            bool status = model.OpenFile(
                ModelPath("pose_landmark_heavy.onnx.prototxt"),
                ModelPath("pose_landmark_heavy.onnx"));
            if (!status)
            {
                Console.WriteLine($"OpenFile failed: {model.GetErrorDetail()}");
                Assert.Ignore("Could not open estimation model (ailia native library may not be available)");
            }

            // Input: NHWC [1, 256, 256, 3] -> AILIAShape: x=3, y=256, z=256, w=1
            int inputIdx = model.FindBlobIndexByName("input_1");
            Assert.That(inputIdx, Is.GreaterThanOrEqualTo(0), "input_1 blob found");

            // This is the same SetInputBlobShape call that causes the error in Unity
            status = model.SetInputBlobShape(new Ailia.AILIAShape
            {
                x = 3, y = 256, z = 256, w = 1, dim = 4
            }, inputIdx);
            Console.WriteLine($"SetInputBlobShape result: {status}");
            if (!status)
            {
                Console.WriteLine($"SetInputBlobShape error: {model.GetErrorDetail()}");
            }
            Assert.That(status, Is.True, $"SetInputBlobShape: {model.GetErrorDetail()}");

            float[] input = new float[1 * 256 * 256 * 3];
            for (int i = 0; i < input.Length; i++)
                input[i] = 0.5f;

            status = model.SetInputBlobData(input, inputIdx);
            Assert.That(status, Is.True, $"SetInputBlobData: {model.GetErrorDetail()}");

            // This Update() call is where the -7 error occurs in Unity
            status = model.Update();
            Console.WriteLine($"Update result: {status}");
            if (!status)
            {
                string error = model.GetErrorDetail();
                Console.WriteLine($"Update error: {error}");
                Assert.Fail($"Update failed (this reproduces the Unity -7 error): {error}");
            }

            // 5 outputs: Identity[1,195], Identity_1[1,1], Identity_2[1,128,128,1],
            //            Identity_3[1,64,64,39], Identity_4[1,117]
            int landmarkIdx = model.FindBlobIndexByName("Identity");
            int scoreIdx = model.FindBlobIndexByName("Identity_1");

            float[] landmarks = new float[195];
            float[] score = new float[1];

            status = model.GetBlobData(score, scoreIdx);
            Assert.That(status, Is.True, $"GetBlobData score: {model.GetErrorDetail()}");

            status = model.GetBlobData(landmarks, landmarkIdx);
            Assert.That(status, Is.True, $"GetBlobData landmarks: {model.GetErrorDetail()}");

            Console.WriteLine($"Estimation model inference succeeded");
            Console.WriteLine($"Pose score (raw): {score[0]:F4}");
            Console.WriteLine($"Landmarks range: [{landmarks.Min():F4}, {landmarks.Max():F4}]");

            // Verify landmarks can be decoded
            for (int i = 0; i < 33; i++)
            {
                float x = landmarks[i * 5] / 256f;
                float y = landmarks[i * 5 + 1] / 256f;
                Console.WriteLine($"  Landmark {i}: ({x:F3}, {y:F3})");
            }
        }
        catch (DllNotFoundException ex)
        {
            Assert.Ignore($"ailia native library not found: {ex.Message}");
        }
        finally
        {
            model.Close();
        }
    }

    // =======================================================
    // 3. Full pipeline: detection + estimation
    // =======================================================
    [Test]
    public void FullPipeline_DetectionAndEstimation()
    {
        if (!TryDownloadModels(DETECTION_FILES) || !TryDownloadModels(ESTIMATION_FILES))
            Assert.Ignore("Could not download models");

        AiliaModel detection = new AiliaModel();
        AiliaModel estimation = new AiliaModel();

        try
        {
            bool status = detection.OpenFile(
                ModelPath("pose_detection.onnx.prototxt"),
                ModelPath("pose_detection.onnx"));
            if (!status)
                Assert.Ignore("Could not open detection model");

            status = estimation.OpenFile(
                ModelPath("pose_landmark_heavy.onnx.prototxt"),
                ModelPath("pose_landmark_heavy.onnx"));
            if (!status)
                Assert.Ignore("Could not open estimation model");

            // Step 1: Run detection
            int detInputIdx = detection.FindBlobIndexByName("input_1");
            status = detection.SetInputBlobShape(new Ailia.AILIAShape
            {
                x = 3, y = 224, z = 224, w = 1, dim = 4
            }, detInputIdx);
            Assert.That(status, Is.True, "Detection SetInputBlobShape");

            float[] detInput = new float[1 * 224 * 224 * 3];
            for (int i = 0; i < detInput.Length; i++)
                detInput[i] = 0.5f;

            status = detection.SetInputBlobData(detInput, detInputIdx);
            Assert.That(status, Is.True, "Detection SetInputBlobData");

            status = detection.Update();
            Assert.That(status, Is.True, $"Detection Update: {detection.GetErrorDetail()}");

            Console.WriteLine("Detection model Update succeeded");

            // Step 2: Run estimation
            int estInputIdx = estimation.FindBlobIndexByName("input_1");
            status = estimation.SetInputBlobShape(new Ailia.AILIAShape
            {
                x = 3, y = 256, z = 256, w = 1, dim = 4
            }, estInputIdx);
            Assert.That(status, Is.True, $"Estimation SetInputBlobShape: {estimation.GetErrorDetail()}");

            float[] estInput = new float[1 * 256 * 256 * 3];
            for (int i = 0; i < estInput.Length; i++)
                estInput[i] = 0.5f;

            status = estimation.SetInputBlobData(estInput, estInputIdx);
            Assert.That(status, Is.True, "Estimation SetInputBlobData");

            status = estimation.Update();
            Assert.That(status, Is.True, $"Estimation Update: {estimation.GetErrorDetail()}");

            Console.WriteLine("Estimation model Update succeeded");
            Console.WriteLine("Full pipeline passed: both models infer successfully with blob API");
        }
        catch (DllNotFoundException ex)
        {
            Assert.Ignore($"ailia native library not found: {ex.Message}");
        }
        finally
        {
            detection.Close();
            estimation.Close();
        }
    }
}
