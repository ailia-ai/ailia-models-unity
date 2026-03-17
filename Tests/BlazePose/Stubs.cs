/* Unity type stubs for non-Unity builds (standalone tests) */
/* These stubs provide minimal implementations of UnityEngine types so that
 * AiliaBlazepose.cs can be compiled and tested outside of Unity.
 * ailia SDK types (AiliaModel, Ailia, AiliaPoseEstimator) are provided by
 * the real ailia-csharp SDK linked in the project file. */

namespace UnityEngine
{
    public class Debug
    {
        public static void Log(object text) { System.Console.WriteLine(text); }
        public static void LogError(object text) { System.Console.WriteLine(text); }
        public static void LogWarning(object text) { System.Console.WriteLine(text); }
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct Color32
    {
        public byte r, g, b, a;
        public Color32(byte r, byte g, byte b, byte a) { this.r = r; this.g = g; this.b = b; this.a = a; }
    }

    public struct Vector2
    {
        public float x, y;
        public Vector2(float x, float y) { this.x = x; this.y = y; }
        public static Vector2 operator *(Vector2 a, float d) => new Vector2(a.x * d, a.y * d);
        public static Vector2 operator *(float d, Vector2 a) => new Vector2(a.x * d, a.y * d);
        public static Vector2 operator +(Vector2 a, Vector2 b) => new Vector2(a.x + b.x, a.y + b.y);
        public static Vector2 operator /(Vector2 a, float d) => new Vector2(a.x / d, a.y / d);
        public static Vector2 one => new Vector2(1, 1);
    }

    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static Vector3 operator +(Vector3 a, Vector3 b) => new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
        public static Vector3 operator /(Vector3 a, float d) => new Vector3(a.x / d, a.y / d, a.z / d);
        public static Vector3 zero => new Vector3(0, 0, 0);
    }

    public static class Mathf
    {
        public const float PI = (float)System.Math.PI;
        public static float Exp(float x) => (float)System.Math.Exp(x);
        public static float Sqrt(float x) => (float)System.Math.Sqrt(x);
        public static float Pow(float x, float y) => (float)System.Math.Pow(x, y);
        public static float Cos(float x) => (float)System.Math.Cos(x);
        public static float Sin(float x) => (float)System.Math.Sin(x);
        public static float Atan2(float y, float x) => (float)System.Math.Atan2(y, x);
        public static float Max(float a, float b) => System.Math.Max(a, b);
        public static float Min(float a, float b) => System.Math.Min(a, b);
        public static float Clamp(float value, float min, float max) => System.Math.Max(min, System.Math.Min(max, value));
    }

    public static class JsonUtility
    {
        public static T FromJson<T>(string json) => System.Text.Json.JsonSerializer.Deserialize<T>(json);
    }

    public static class Shader
    {
        public static int PropertyToID(string name) => name.GetHashCode();
    }

    public enum SystemLanguage
    {
        Japanese, Chinese, ChineseSimplified, ChineseTraditional, English
    }

    public static class Application
    {
        public static SystemLanguage systemLanguage => SystemLanguage.English;
    }
}

// ailiaSDK namespace: AiliaBlazepose.cs uses "using ailiaSDK;" but the real
// ailia SDK defines AiliaPoseEstimator in the "ailia" namespace.
// Provide an empty namespace so the using directive compiles.
namespace ailiaSDK { }
