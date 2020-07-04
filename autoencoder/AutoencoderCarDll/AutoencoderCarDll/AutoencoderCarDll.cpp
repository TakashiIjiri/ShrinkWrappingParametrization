// AutoencoderCarDll.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。


#include "stdafx.h"
#include "AutoencoderForCarDll.h"

#include <iostream>



// 準備 : 
// 1: python + pythorch側で学習して重みファイルを作成 
//    autoencoder_shrinkwraping.pyを参照
//    autoencoderを学習し，saveする
//
// 2: 重みを読み出してモデル自体をセーブ（.ptファイル）
//    モデルの一部のみを実行し，その部分だけをセーブする機能がある
//    autoencoder_shrinkwraping_convert_pt.pyの以下の部分を参照
//    
// example = torch.rand(1, N) # Nはオートエンコーダの中間層サイズ
// traced_net = torch.jit.trace(model.decoder, example)
// traced_net.save("decoder.pt")
//

// Libtorchをダウンロードして展開
//  cuda10.1, windowsを指定（cuda使っていないので不要だったかも）
//
//以下のサイトを参考にした
//https://medium.com/@boonboontongbuasirilai/building-pytorch-c-integration-libtorch-with-ms-visual-studio-2017-44281f9921ea
//
// プロジェクトの設定「言語」より準拠モードをnonに
// 追加のインクルードディレクトリに以下を追加
// ./3rdparty/libtorch/include;
// ./3rdparty/libtorch/include/torch/csrc;%(AdditionalIncludeDirectories)
//
//追加のライブラリディレクトリに以下を追加
// ./3rdparty/libtorch/lib
//
//追加の依存ファイルに以下を追加
// torch_cpu.lib; c10.lib
//必要なdllをexeと同じ場所へ移動
//
// include "windows.h"をstdafxからdllmainへ移動(大切)
//
// この状態で、下のコードのように，torch::jit::script::Module を利用するとOK


static torch::jit::script::Module decoder;
static torch::jit::script::Module encoder;
static torch::jit::script::Module encoder_decoder;

// ウィンドウの各部の色を得る。
extern "C" __declspec(dllexport) void InitializeAutoencoderCar(
  const char* encoder_decoder_pt_path, 
  const char* encoder_pt_path, 
  const char* decoder_pt_path)
{
  std::cout << "InitializeAutoencoderCar\n";
  //for (auto m : deco.get_methods() ) 
  //  std::cout << m.name();

  encoder_decoder = torch::jit::load( encoder_decoder_pt_path );
  encoder         = torch::jit::load( encoder_pt_path );
  decoder         = torch::jit::load( decoder_pt_path );
}



extern "C" __declspec(dllexport) void AutoencoderCar_EncoderDecoder(float *input, float *output )
{
  std::cout << AutoencoderCar_Encoder << "\n";
  at::Tensor in  = torch::from_blob(input, {1, AE_INPUT_SIZE});
  at::Tensor out = encoder_decoder.forward({in}).toTensor();

  std::cout << in .size(0) << "  " << in .size(1) << "\n";
  std::cout << out.size(0) << "  " << out.size(1) << "\n";

  const float* output_ptr = out.data_ptr<float>();
  memcpy(output, output_ptr, sizeof(float) * AE_INPUT_SIZE );  
}



extern "C" __declspec(dllexport) void AutoencoderCar_Encoder(float *input, float *output )
{
  std::cout << AutoencoderCar_Encoder << "\n";
  at::Tensor in = torch::from_blob(input, {1, AE_INPUT_SIZE});
  at::Tensor out= encoder.forward({in}).toTensor();

  std::cout << in .size(0) << "  " << in .size(1) << "\n";
  std::cout << out.size(0) << "  " << out.size(1) << "\n";

  const float* output_ptr = out.data_ptr<float>();
  memcpy(output, output_ptr, sizeof(float) * AE_MID_SIZE );  
}



extern "C" __declspec(dllexport) void AutoencoderCar_Decode(float *input, float *output )
{
  std::cout << AutoencoderCar_Decode << "\n";
  at::Tensor in  = torch::from_blob(input, {1,AE_MID_SIZE});
  at::Tensor out = decoder.forward({in}).toTensor();

  const float* output_ptr = out.data_ptr<float>();
  std::cout << in .size(0) << "  " << in .size(1) << "\n";
  std::cout << out.size(0) << "  " << out.size(1) << "\n";
  memcpy( output, output_ptr, sizeof(float) * AE_INPUT_SIZE );  
}



