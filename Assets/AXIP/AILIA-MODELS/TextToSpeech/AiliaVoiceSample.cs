using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using System.Threading;
using System.Threading.Tasks;

using ailia;
using ailiaVoice;

namespace ailiaSDK{

public class AiliaVoiceSample : MonoBehaviour
{
	// Model list
	public enum TextToSpeechSampleModels
	{
		tacotron2_english,
		gpt_sovits_japanese,
		gpt_sovits_english,
		gpt_sovits_chinese,
		gpt_sovits_v2_japanese,
		gpt_sovits_v2_english,
		gpt_sovits_v2_chinese,
		gpt_sovits_v3_japanese,
		gpt_sovits_v3_english,
		gpt_sovits_v3_chinese,
		gpt_sovits_v2_pro_japanese,
		gpt_sovits_v2_pro_english,
		gpt_sovits_v2_pro_chinese,
		gpt_sovits_v2_pro_distill_japanese
	}

	// Settings
	public TextToSpeechSampleModels modelType = TextToSpeechSampleModels.gpt_sovits_japanese;
	public GameObject UICanvas = null;

	public AudioClip clip;
	public AudioClip ref_clip;
	public AudioSource audioSource;
	public GameObject processing;
	public bool gpu_mode = false;
	public InputField input_field;
	private string queue_text = "";
	private bool initialized = false;
	private AiliaVoiceModel voice = new AiliaVoiceModel();
	private string before_ref_clip_name = "";
	private bool model_downloading = false;
	private bool use_user_dict = false;
    private Task t = null;

	// model download
	private AiliaDownload ailia_download = new AiliaDownload();


	private bool isProcessing = false;

	// Version helpers
	private bool IsV1(){
		return modelType == TextToSpeechSampleModels.gpt_sovits_japanese || modelType == TextToSpeechSampleModels.gpt_sovits_english || modelType == TextToSpeechSampleModels.gpt_sovits_chinese;
	}
	private bool IsV2(){
		return modelType == TextToSpeechSampleModels.gpt_sovits_v2_japanese || modelType == TextToSpeechSampleModels.gpt_sovits_v2_english || modelType == TextToSpeechSampleModels.gpt_sovits_v2_chinese;
	}
	private bool IsV3(){
		return modelType == TextToSpeechSampleModels.gpt_sovits_v3_japanese || modelType == TextToSpeechSampleModels.gpt_sovits_v3_english || modelType == TextToSpeechSampleModels.gpt_sovits_v3_chinese;
	}
	private bool IsV2Pro(){
		return modelType == TextToSpeechSampleModels.gpt_sovits_v2_pro_japanese || modelType == TextToSpeechSampleModels.gpt_sovits_v2_pro_english || modelType == TextToSpeechSampleModels.gpt_sovits_v2_pro_chinese || modelType == TextToSpeechSampleModels.gpt_sovits_v2_pro_distill_japanese;
	}
	private bool IsV2ProDistill(){
		return modelType == TextToSpeechSampleModels.gpt_sovits_v2_pro_distill_japanese;
	}
	private bool IsJapanese(){
		return modelType == TextToSpeechSampleModels.gpt_sovits_japanese || modelType == TextToSpeechSampleModels.gpt_sovits_v2_japanese || modelType == TextToSpeechSampleModels.gpt_sovits_v3_japanese || modelType == TextToSpeechSampleModels.gpt_sovits_v2_pro_japanese || modelType == TextToSpeechSampleModels.gpt_sovits_v2_pro_distill_japanese;
	}
	private bool IsEnglish(){
		return modelType == TextToSpeechSampleModels.gpt_sovits_english || modelType == TextToSpeechSampleModels.gpt_sovits_v2_english || modelType == TextToSpeechSampleModels.gpt_sovits_v3_english || modelType == TextToSpeechSampleModels.gpt_sovits_v2_pro_english;
	}
	private bool IsChinese(){
		return modelType == TextToSpeechSampleModels.gpt_sovits_chinese || modelType == TextToSpeechSampleModels.gpt_sovits_v2_chinese || modelType == TextToSpeechSampleModels.gpt_sovits_v3_chinese || modelType == TextToSpeechSampleModels.gpt_sovits_v2_pro_chinese;
	}
	private bool IsGPTSoVITS(){
		return IsV1() || IsV2() || IsV3() || IsV2Pro();
	}

	// Start is called before the first frame update
	void Start()
	{
			AiliaLicense.CheckAndDownloadLicense();
		UISetup();
		LoadModel();
	}

	void UISetup()
	{
		Debug.Assert (UICanvas != null, "UICanvas is null");

		Text label_text = UICanvas.transform.Find("LabelText").GetComponent<Text>();
		label_text.text = "";

		Text mode_text = UICanvas.transform.Find("ModeLabel").GetComponent<Text>();
		mode_text.text = "ailia Voice Synthesis Sample";

		UICanvas.transform.Find("RawImage").GetComponent<RawImage>().gameObject.SetActive(false);
	}

	void LoadModel(){
		if (initialized){
			return;
		}

		int env_id = voice.GetEnvironmentId(gpu_mode);
		bool status = voice.Create(env_id, AiliaVoice.AILIA_VOICE_FLAG_NONE);
		if (status == false){
			Debug.Log("Create failed");
			return;
		}

		string asset_path=Application.temporaryCachePath;
		string path = asset_path+"/";

		var urlList = new List<ModelDownloadURL>();

		if (IsGPTSoVITS()){
			urlList.Add(new ModelDownloadURL() { folder_path = "open_jtalk/open_jtalk_dic_utf_8-1.11", file_name = "char.bin" });
			urlList.Add(new ModelDownloadURL() { folder_path = "open_jtalk/open_jtalk_dic_utf_8-1.11", file_name = "COPYING" });
			urlList.Add(new ModelDownloadURL() { folder_path = "open_jtalk/open_jtalk_dic_utf_8-1.11", file_name = "left-id.def" });
			urlList.Add(new ModelDownloadURL() { folder_path = "open_jtalk/open_jtalk_dic_utf_8-1.11", file_name = "matrix.bin" });
			urlList.Add(new ModelDownloadURL() { folder_path = "open_jtalk/open_jtalk_dic_utf_8-1.11", file_name = "pos-id.def" });
			urlList.Add(new ModelDownloadURL() { folder_path = "open_jtalk/open_jtalk_dic_utf_8-1.11", file_name = "rewrite.def" });
			urlList.Add(new ModelDownloadURL() { folder_path = "open_jtalk/open_jtalk_dic_utf_8-1.11", file_name = "right-id.def" });
			urlList.Add(new ModelDownloadURL() { folder_path = "open_jtalk/open_jtalk_dic_utf_8-1.11", file_name = "sys.dic" });
			urlList.Add(new ModelDownloadURL() { folder_path = "open_jtalk/open_jtalk_dic_utf_8-1.11", file_name = "unk.dic" });

			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "user.dict" });
		}
		if (IsEnglish()){
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_en", file_name = "averaged_perceptron_tagger_classes.txt" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_en", file_name = "averaged_perceptron_tagger_tagdict.txt" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_en", file_name = "averaged_perceptron_tagger_weights.txt" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_en", file_name = "cmudict" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_en", file_name = "g2p_decoder.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_en", file_name = "g2p_encoder.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_en", file_name = "homographs.en" });
		}
		if (IsChinese()){
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_cn", file_name = "pinyin.txt" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_cn", file_name = "opencpop-strict.txt" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_cn", file_name = "jieba.dict.utf8" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_cn", file_name = "hmm_model.utf8" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_cn", file_name = "user.dict.utf8" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_cn", file_name = "idf.utf8" });
			urlList.Add(new ModelDownloadURL() { folder_path = "g2p_cn", file_name = "stop_words.utf8" });
			if (IsV2() || IsV3() || IsV2Pro()){
				urlList.Add(new ModelDownloadURL() { folder_path = "g2pw/1.1", file_name = "g2pW.onnx" });
				urlList.Add(new ModelDownloadURL() { folder_path = "g2pw/1.1", file_name = "POLYPHONIC_CHARS.txt" });
				urlList.Add(new ModelDownloadURL() { folder_path = "g2pw/1.1", file_name = "bopomofo_to_pinyin_wo_tune_dict.json" });
				urlList.Add(new ModelDownloadURL() { folder_path = "g2pw/gpt-sovits", file_name = "polyphonic.rep" });
				urlList.Add(new ModelDownloadURL() { folder_path = "g2pw/gpt-sovits", file_name = "polyphonic-fix.rep" });
			}
		}
		if (modelType == TextToSpeechSampleModels.tacotron2_english){
			urlList.Add(new ModelDownloadURL() { folder_path = "tacotron2", file_name = "encoder.onnx", local_name = "nivdia_encoder.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "tacotron2", file_name = "decoder_iter.onnx", local_name = "nivdia_decoder_iter.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "tacotron2", file_name = "postnet.onnx", local_name = "nivdia_postnet.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "tacotron2", file_name = "waveglow.onnx", local_name = "nivdia_waveglow.onnx" });
		}
		if (IsV1()){
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits", file_name = "t2s_encoder.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits", file_name = "t2s_fsdec.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits", file_name = "t2s_sdec.opt3.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits", file_name = "vits.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits", file_name = "cnhubert.onnx" });
		}
		if (IsV2()){
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2", file_name = "t2s_encoder.onnx", local_name = "v2_t2s_encoder.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2", file_name = "t2s_fsdec.onnx", local_name = "v2_t2s_fsdec.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2", file_name = "t2s_sdec.opt.onnx", local_name = "v2_t2s_sdec.opt.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2", file_name = "vits.onnx", local_name = "v2_vits.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2", file_name = "cnhubert.onnx", local_name = "v2_cnhubert.onnx" });
			if (IsChinese()){
				urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2", file_name = "chinese-roberta.onnx" });
				urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2", file_name = "vocab.txt" });
			}
		}
		if (IsV3()){
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "t2s_encoder.onnx", local_name = "v3_t2s_encoder.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "t2s_fsdec.onnx", local_name = "v3_t2s_fsdec.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "t2s_sdec.opt.onnx", local_name = "v3_t2s_sdec.opt.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "cnhubert.onnx", local_name = "v3_cnhubert.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "vq_model.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "vq_cfm.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "bigvgan_model.onnx" });
			if (IsChinese()){
				urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "chinese-roberta.onnx" });
				urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v3", file_name = "vocab.txt" });
			}
		}
		if (IsV2Pro() && !IsV2ProDistill()){
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "t2s_encoder.onnx", local_name = "v2pro_t2s_encoder.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "t2s_fsdec.onnx", local_name = "v2pro_t2s_fsdec.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "t2s_sdec.opt.onnx", local_name = "v2pro_t2s_sdec.opt.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "cnhubert.onnx", local_name = "v2pro_cnhubert.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "vits.onnx", local_name = "v2pro_vits.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "sv.onnx", local_name = "v2pro_sv.onnx" });
			if (IsChinese()){
				urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "chinese-roberta.onnx" });
				urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "vocab.txt" });
			}
		}
		if (IsV2ProDistill()){
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro-distill", file_name = "t2s_encoder_distill_small.onnx", local_name = "v2pro_distill_t2s_encoder.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro-distill", file_name = "t2s_fsdec_distill_small.onnx", local_name = "v2pro_distill_t2s_fsdec.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro-distill", file_name = "t2s_sdec_distill_small.opt.onnx", local_name = "v2pro_distill_t2s_sdec.opt.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "cnhubert.onnx", local_name = "v2pro_cnhubert.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "vits.onnx", local_name = "v2pro_vits.onnx" });
			urlList.Add(new ModelDownloadURL() { folder_path = "gpt-sovits-v2-pro", file_name = "sv.onnx", local_name = "v2pro_sv.onnx" });
		}

		AiliaDownload ailia_download = new AiliaDownload();
		ailia_download.DownloaderProgressPanel = UICanvas.transform.Find("DownloaderProgressPanel").gameObject;

		StartCoroutine(ailia_download.DownloadWithProgressFromURL(urlList, () =>
		{
			if (IsGPTSoVITS()){
				status = voice.SetUserDictionary(path + "user.dict", AiliaVoice.AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK);
				if (status == false) {
					Debug.Log("SetUserDictionary failed");
					return;
				}
				use_user_dict = true;
				status = voice.OpenDictionary(path, AiliaVoice.AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK);
				if (status == false){
					Debug.Log("OpenDictionary failed");
					return;
				}
			}
			if (IsEnglish()){
				status = voice.OpenDictionary(path, AiliaVoice.AILIA_VOICE_DICTIONARY_TYPE_G2P_EN);
				if (status == false){
					Debug.Log("OpenDictionary G2P_EN failed");
					return;
				}
			}
			if (IsChinese()){
				status = voice.OpenDictionary(path, AiliaVoice.AILIA_VOICE_DICTIONARY_TYPE_G2P_CN);
				if (status == false){
					Debug.Log("OpenDictionary G2P_CN failed");
					return;
				}
				if (IsV2() || IsV3() || IsV2Pro()){
					status = voice.OpenDictionary(path, AiliaVoice.AILIA_VOICE_DICTIONARY_TYPE_G2PW);
					if (status == false){
						Debug.Log("OpenDictionary G2PW failed");
						return;
					}
				}
			}

			switch(modelType){
			case TextToSpeechSampleModels.tacotron2_english:
				Debug.Log(path+"nivdia_encoder.onnx");
				status = voice.OpenModel(path+"nivdia_encoder.onnx", path+"nivdia_decoder_iter.onnx", path+"nivdia_postnet.onnx", path+"nivdia_waveglow.onnx", null, AiliaVoice.AILIA_VOICE_MODEL_TYPE_TACOTRON2, AiliaVoice.AILIA_VOICE_CLEANER_TYPE_BASIC);
				break;
			case TextToSpeechSampleModels.gpt_sovits_japanese:
			case TextToSpeechSampleModels.gpt_sovits_english:
			case TextToSpeechSampleModels.gpt_sovits_chinese:
				status = voice.OpenGPTSoVITSV1ModelFile(path+"t2s_encoder.onnx", path+"t2s_fsdec.onnx", path+"t2s_sdec.opt3.onnx", path+"vits.onnx", path+"cnhubert.onnx");
				break;
			case TextToSpeechSampleModels.gpt_sovits_v2_japanese:
			case TextToSpeechSampleModels.gpt_sovits_v2_english:
				status = voice.OpenGPTSoVITSV2ModelFile(path+"v2_t2s_encoder.onnx", path+"v2_t2s_fsdec.onnx", path+"v2_t2s_sdec.opt.onnx", path+"v2_vits.onnx", path+"v2_cnhubert.onnx", null, null);
				break;
			case TextToSpeechSampleModels.gpt_sovits_v2_chinese:
				status = voice.OpenGPTSoVITSV2ModelFile(path+"v2_t2s_encoder.onnx", path+"v2_t2s_fsdec.onnx", path+"v2_t2s_sdec.opt.onnx", path+"v2_vits.onnx", path+"v2_cnhubert.onnx", path+"chinese-roberta.onnx", path+"vocab.txt");
				break;
			case TextToSpeechSampleModels.gpt_sovits_v3_japanese:
			case TextToSpeechSampleModels.gpt_sovits_v3_english:
				status = voice.OpenGPTSoVITSV3ModelFile(path+"v3_t2s_encoder.onnx", path+"v3_t2s_fsdec.onnx", path+"v3_t2s_sdec.opt.onnx", path+"v3_cnhubert.onnx", path+"vq_model.onnx", path+"vq_cfm.onnx", path+"bigvgan_model.onnx", null, null);
				break;
			case TextToSpeechSampleModels.gpt_sovits_v3_chinese:
				status = voice.OpenGPTSoVITSV3ModelFile(path+"v3_t2s_encoder.onnx", path+"v3_t2s_fsdec.onnx", path+"v3_t2s_sdec.opt.onnx", path+"v3_cnhubert.onnx", path+"vq_model.onnx", path+"vq_cfm.onnx", path+"bigvgan_model.onnx", path+"chinese-roberta.onnx", path+"vocab.txt");
				break;
			case TextToSpeechSampleModels.gpt_sovits_v2_pro_japanese:
			case TextToSpeechSampleModels.gpt_sovits_v2_pro_english:
				status = voice.OpenGPTSoVITSV2ProModelFile(path+"v2pro_t2s_encoder.onnx", path+"v2pro_t2s_fsdec.onnx", path+"v2pro_t2s_sdec.opt.onnx", path+"v2pro_cnhubert.onnx", path+"v2pro_vits.onnx", path+"v2pro_sv.onnx", null, null);
				break;
			case TextToSpeechSampleModels.gpt_sovits_v2_pro_chinese:
				status = voice.OpenGPTSoVITSV2ProModelFile(path+"v2pro_t2s_encoder.onnx", path+"v2pro_t2s_fsdec.onnx", path+"v2pro_t2s_sdec.opt.onnx", path+"v2pro_cnhubert.onnx", path+"v2pro_vits.onnx", path+"v2pro_sv.onnx", path+"chinese-roberta.onnx", path+"vocab.txt");
				break;
			case TextToSpeechSampleModels.gpt_sovits_v2_pro_distill_japanese:
				status = voice.OpenGPTSoVITSV2ProModelFile(path+"v2pro_distill_t2s_encoder.onnx", path+"v2pro_distill_t2s_fsdec.onnx", path+"v2pro_distill_t2s_sdec.opt.onnx", path+"v2pro_cnhubert.onnx", path+"v2pro_vits.onnx", path+"v2pro_sv.onnx", null, null);
				break;
			}
			if (status == false){
				Debug.Log("OpenModel failed");
				return;
			}
			initialized = true;
			before_ref_clip_name = "";
		}));
	}

	private int GetG2PType(){
		if (IsJapanese()) return AiliaVoice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA;
		if (IsEnglish()) return AiliaVoice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN;
		if (IsChinese()) return AiliaVoice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_ZH;
		return AiliaVoice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA;
	}

	private void Infer(string text){
		if (IsGPTSoVITS()){
			if (ref_clip.name != before_ref_clip_name){
				string label = "水をマレーシアから買わなくてはならない。";
				Debug.Log("Label : " + label);
				string ref_text = voice.G2P(label, AiliaVoice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA);
				voice.SetReference(ref_clip, ref_text);
				before_ref_clip_name = ref_clip.name;
			}
		}
		if (IsGPTSoVITS()){
			if (use_user_dict){
				text = text.ToLower();
			}
			text = voice.G2P(text, GetG2PType());
		}

		Debug.Log("Features : "+ text);
		if (text == "") {
			return;
		}

		var context = SynchronizationContext.Current;

		t = Task.Run(async () =>
		{
			isProcessing = true;

			bool status = voice.Inference(text);
			if (status == false){
				Debug.Log("Inference error");
				return;
			}

			context.Post(state =>
			{
				clip = voice.GetAudioClip();
				if (status == null){
					Debug.Log("Inference failed");
					isProcessing = false;
					return;
				}

				Debug.Log("Samples : " + clip.samples);
				Debug.Log("Channels : " + clip.channels);
				Debug.Log("SamplingRate : " + clip.frequency);

				audioSource.clip = clip;
				audioSource.Play();

				isProcessing = false;
			}, null);
		});//.Wait();
	}

	void OnDisable(){
		Debug.Log("OnDisable");
		if (t != null){
			t.Wait();
		}
		voice.Close();
		initialized = false;
	}

	void OnApplicationQuit(){
		Debug.Log("OnApplicationQuit");
		if (t != null){
			t.Wait();
		}
		voice.Close();
		initialized = false;
	}

	// Update is called once per frame
	void Update()
	{
		if (isProcessing){
			processing.SetActive(true);
			return;
		}else{
			processing.SetActive(false);
		}
		if (queue_text != ""){
			if (initialized){
				Infer(queue_text);
				queue_text = "";
			}
		}
	}

	public void Speak(){
		if (input_field.text == "") {
			return;
		}
		queue_text = "" + input_field.text;
		Debug.Log("Queue : " + queue_text);
		input_field.text = "";
	}

	public void Replay(){
		audioSource.Play();
	}
}

}
