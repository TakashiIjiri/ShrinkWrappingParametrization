#pragma once


#define AE_INPUT_SIZE 6162
#define AE_MID_SIZE   20

extern "C" __declspec(dllexport) void InitializeAutoencoderCar(
  const char* encoder_decoder_pt_path, 
  const char* encoder_pt_path, 
  const char* decoder_pt_path);

extern "C" __declspec(dllexport) void AutoencoderCar_EncoderDecoder(float *input, float *output);
extern "C" __declspec(dllexport) void AutoencoderCar_Encoder(float *input, float *output);
extern "C" __declspec(dllexport) void AutoencoderCar_Decode(float *input, float *output);



