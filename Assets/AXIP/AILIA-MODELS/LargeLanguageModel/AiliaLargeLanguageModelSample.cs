/* AILIA Unity Plugin Large Language Model Sample */
/* Copyright 2024 AXELL CORPORATION */

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using ailia;
using ailiaLLM;

namespace ailiaSDK
{
	public class AiliaLargeLanguageModelSample : MonoBehaviour
	{
		// Model list
		public enum LargeLanguageModelSampleModels
		{
			gemma2_2b,
			gemma3_1b,
			gemma3_4b,
			gemma3n_E2B,
			gemma3n_E4B,
			gemma4_E2B,
			gemma4_E4B,
			llama3_2_3b,
		}

		// UI
		[SerializeField]
		public InputField input_field;

		// Settings
		public LargeLanguageModelSampleModels modelType = LargeLanguageModelSampleModels.gemma2_2b;
		public bool gpu_mode = false;
		public int n_ctx = 8192;
		public GameObject UICanvas = null;

		// Result
		Text label_text = null;
		Text mode_text = null;

		// AILIA
		private AiliaLLMModel llm = null;
		private List<AiliaLLMChatMessage> messages = new List<AiliaLLMChatMessage>(); // Chat History

		bool modelPrepared = false;
		bool modelAllocated = false;
		bool done = true;
		string generate_text = "";

		void Start()
		{
			AiliaLicense.CheckAndDownloadLicense();
			UISetup();

			// for Processing
			AiliaInit();
		}

		void UISetup()
		{
			Debug.Assert (UICanvas != null, "UICanvas is null");

			label_text = UICanvas.transform.Find("LabelText").GetComponent<Text>();
			mode_text = UICanvas.transform.Find("ModeLabel").GetComponent<Text>();

			mode_text.text = "ailia Large Language Model Sample";
			label_text.text = "Please input query.";

			UICanvas.transform.Find("RawImage").GetComponent<RawImage>().gameObject.SetActive(false);
		}

		void AiliaInit()
		{
			// Create Ailia
			CreateAiliaNet(modelType, gpu_mode);
		}

		// Download models and Create ailiaModel
		void CreateAiliaNet(LargeLanguageModelSampleModels modelType, bool gpu_mode = true)
		{
			string asset_path = Application.temporaryCachePath;

			AiliaDownload ailia_download = new AiliaDownload();
			ailia_download.DownloaderProgressPanel = UICanvas.transform.Find("DownloaderProgressPanel").gameObject;
			var urlList = new List<ModelDownloadURL>();

			if (modelType == LargeLanguageModelSampleModels.gemma2_2b){
				urlList.Add(new ModelDownloadURL() { folder_path = "gemma", file_name = "gemma-2-2b-it-Q4_K_M.gguf" });
			}
			if (modelType == LargeLanguageModelSampleModels.gemma3_1b){
				urlList.Add(new ModelDownloadURL() { folder_path = "gemma", file_name = "gemma-3-1b-it-Q4_K_M.gguf" });
			}
			if (modelType == LargeLanguageModelSampleModels.gemma3_4b){
				urlList.Add(new ModelDownloadURL() { folder_path = "gemma", file_name = "gemma-3-4b-it-Q4_K_M.gguf" });
			}
			if (modelType == LargeLanguageModelSampleModels.gemma3n_E2B){
				urlList.Add(new ModelDownloadURL() { folder_path = "gemma", file_name = "gemma-3n-E2B-it-Q4_K_M.gguf" });
			}
			if (modelType == LargeLanguageModelSampleModels.gemma3n_E4B){
				urlList.Add(new ModelDownloadURL() { folder_path = "gemma", file_name = "gemma-3n-E4B-it-Q4_K_M.gguf" });
			}
			if (modelType == LargeLanguageModelSampleModels.gemma4_E2B){
				urlList.Add(new ModelDownloadURL() { folder_path = "gemma", file_name = "gemma-4-E2B-it-Q4_K_M.gguf" });
			}
			if (modelType == LargeLanguageModelSampleModels.gemma4_E4B){
				urlList.Add(new ModelDownloadURL() { folder_path = "gemma", file_name = "gemma-4-E4B-it-Q4_K_M.gguf" });
			}
			if (modelType == LargeLanguageModelSampleModels.llama3_2_3b){
				urlList.Add(new ModelDownloadURL() { folder_path = "llama-3", file_name = "Llama-3.2-3B-Instruct-Q4_K_M.gguf" });
			}

			StartCoroutine(ailia_download.DownloadWithProgressFromURL(urlList, () =>
			{
				llm = new AiliaLLMModel();
				llm.Create();
				if (modelType == LargeLanguageModelSampleModels.gemma2_2b){
					modelPrepared = llm.Open(asset_path + "/gemma-2-2b-it-Q4_K_M.gguf", n_ctx);
				}
				if (modelType == LargeLanguageModelSampleModels.gemma3_1b){
					modelPrepared = llm.Open(asset_path + "/gemma-3-1b-it-Q4_K_M.gguf", n_ctx);
				}
				if (modelType == LargeLanguageModelSampleModels.gemma3_4b){
					modelPrepared = llm.Open(asset_path + "/gemma-3-4b-it-Q4_K_M.gguf", n_ctx);
				}
				if (modelType == LargeLanguageModelSampleModels.gemma3n_E2B){
					modelPrepared = llm.Open(asset_path + "/gemma-3n-E2B-it-Q4_K_M.gguf", n_ctx);
				}
				if (modelType == LargeLanguageModelSampleModels.gemma3n_E4B){
					modelPrepared = llm.Open(asset_path + "/gemma-3n-E4B-it-Q4_K_M.gguf", n_ctx);
				}
				if (modelType == LargeLanguageModelSampleModels.gemma4_E2B){
					modelPrepared = llm.Open(asset_path + "/gemma-4-E2B-it-Q4_K_M.gguf", n_ctx);
				}
				if (modelType == LargeLanguageModelSampleModels.gemma4_E4B){
					modelPrepared = llm.Open(asset_path + "/gemma-4-E4B-it-Q4_K_M.gguf", n_ctx);
				}
				if (modelType == LargeLanguageModelSampleModels.llama3_2_3b){
					modelPrepared = llm.Open(asset_path + "/Llama-3.2-3B-Instruct-Q4_K_M.gguf", n_ctx);
				}
				if (modelPrepared == false){
					Debug.Log("ailiaModel.OpenFile failed");
				}

				SetSystemPrompt();
			}));
		}

		private void SetSystemPrompt(){
			AiliaLLMChatMessage message = new AiliaLLMChatMessage();
			message.role = "system";
			message.content = "あなたは可愛いくまのキャラクターです。発言の語尾に「くま」をつけてください。";
			messages.Add(message);
		}

		void Update()
		{
			if (!modelPrepared)
			{
				return;
			}

			Generate();
		}

		void OnApplicationQuit()
		{
			DestroyAilia();
		}

		void OnDestroy()
		{
			DestroyAilia();
		}

		private void DestroyAilia()
		{
			llm.Close();
		}

		private void Generate(){
			if (done == true){
				return;
			}

			bool status = llm.Generate(ref done);
			if (status == false){
				// context size full
				done = true;
				return;
			}

			if (done == false){
				string deltaText = llm.GetDeltaText();
				generate_text = generate_text + deltaText;
				label_text.text = generate_text;
			}

			if (done){
				AiliaLLMChatMessage message = new AiliaLLMChatMessage();
				message.role = "assistant";
				message.content = generate_text;
				messages.Add(message);
			}
		}

		public void Submit(){
			if (!modelPrepared) {
				return;
			}
			if (done == false) {
				return;
			}

			string query_text = input_field.text;

			if (llm.ContextFull()){
				messages = new List<AiliaLLMChatMessage>();
				SetSystemPrompt();
			}

			AiliaLLMChatMessage message = new AiliaLLMChatMessage();
			message.role = "user";
			message.content = query_text;
			messages.Add(message);

			input_field.text = "";
			generate_text = "";

			llm.SetPrompt(messages);
			done = false;
		}
	}
}
