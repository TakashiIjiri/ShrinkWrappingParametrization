�ȉ��̘_���̒ǎ��p���|�W�g���D  
https://www.autodeskresearch.com/publications/exploring_generative_3d_shapes  
  
    
1. ShrinkWrappingParametrization  
+ �_�����Œ�Ă���Ă���Shrink wrapping parametrization��  
+ Autoencoder��ʂ������f���̉���  
�@���s�Ȃ�C++/CLI�v���W�F�N�g  
  
    
2. autoencoder�t�H���_  
+ autoencoder_shrinkwraping.py : pytorch�ɂ��autoencoder����  
+ autoencoder_shrinkwraping_convert_pt.py: autoencoder�Əd�݂��C.pt�`���ŏ����o��  
+ AutoencoderCarDll : .pt�t�@�C����libtorch�ɂ��dll������v���W�F�N�g  


�ǎ������菇 �̃����i�����p�j

1.�f�[�^����
�@�~�J����̃y�[�W���ihttp://www.nobuyuki-umetani.com/publication/2017_siggatb_explore/2017_siggatb_ExploringGenerative3Dshapes_Objs.zip�j
�@�Ԃ�obj���f�����_�E�����[�h�D  
�@����͎Ԃ�4�p�`���b�V�����f���Q�ŁC�����Shrink wraping parametrization���邱�Ƃœ����x�N�g���ɕϊ�����

2. �f�[�^�̕ϊ� 
  �Ԃ�obj���f�������ׂēW�J���āC�ȉ��̃t�H���_�ɔz�u�D
  ./ShrinkWrappingParametrization/ShrinkWrappingParametrization/models
  ShrinkWrappingParametrization�v���W�F�N�g���R���p�C�����Ď��s����ƁC
  model���̂��ׂĂ�obj���p�����[�^�����ē����t�H���_�Ƀe�L�X�g�P�[���ŕۑ�����D

  �����̎��s�ɂ́Cglew��Eigen���_�E�����[�h����3rdparty�t�H���_�ɔz�u����K�v����
  ��TCore�̃R���X�g���N�^���t�@�C���ϊ����Ăяo���D���݂��̕�����if(false)���Ă���

3. Autoencoder�̊w�K
�@autoencoder_shrinkwraping.py�Ɠ����t�H���_��models�Ƃ����t�H���_�����C2�Ő������ꂽ���ׂĂ� .txt�t�@�C��������
  autoencoder_shrinkwraping.py�����s����Ɗw�K���n�܂�C�w�K�r���̏d�݃f�[�^��
  ./autoencoder_sw  
  �t�H���_�ɂ��܂��Ă����D

  �w�K��conversion.py�����s����ƁC�d�݃f�[�^��ǂݍ��݁C�l�b�g���[�N�Əd�݂�.pt�t�@�C���Ƃ��ăZ�[�u����
  
  �����ԑw�̃��j�b�g����Dropout�̗L���ȂǐF�X�ς��Ď��s���Ă݂�
  �����݂̂Ƃ��냆�j�b�g����20���炢���ƌ��\�ǂ����ʂ��o��
  ��autoencoder_shrinkwraping.py��conversion.py���Q�Ƃ���t�H���_�͓K�X����������K�v����
  �����s�ɂ�pytorch���K�v


4. libtorch�ɂ�� �I�[�g�G���R�[�_��C++�œ�����dll���쐬
�@libtorch���_�E�����[�h�A�𓀂���
�@./autoencoder/AutoencoderCarDll/AutoencoderCarDll/3rdparty
  �ɔz�u���ăR���p�C���D
  ���ׂ��ȃv���W�F�N�g�t�@�C���̐ݒ���@�Ȃǂ́C���̃v���W�F�N�g��AutoencoderCarDll.cpp�ɋL�ځD
  ���x���͑�R�o�邪�Ƃ肠���������B�B�B

  �������ꂽ�ȉ��̃t�@�C����������ׂ��Ƃ���֔z�u
  AutoencoderCarDll.dll  --> ShrinkWrappingParametrization�v���W�F�N�g��exe�Ɠ����ꏊ��
  AutoencoderCarDll.lib  --> ShrinkWrappingParametrization�v���W�F�N�g��mydll��
  AutoencoderForCarDll.h --> ShrinkWrappingParametrization�v���W�F�N�g��mydll��


5. Autoencoder�̐��x�e�X�g�ƁCdecoder�𗘗p�����Ԃ̃����_������
�@2�ŗ��p����ShrinkWrappingParametrization�v���W�F�N�g�𗘗p����D

  ./ShrinkWrappingParametrization/ShrinkWrappingParametrization �t�H���_���ɁC3�ō쐬�����w�K�ς݂̃��f���f�[�^(pt)��z�u�D
  full.pt : Encoder --> Decoder�̃l�b�g���[�N
  encoder.pt : Encoder �݂̂̃l�b�g���[�N
  decoder.pt : Decoder �݂̂̃l�b�g���[�N

  �� "a"�L�[�������ƃ����_���l���I�[�g�G���R�[�_�ɗ^���C�������玩���I�Ɏԃ��f���𐶐�����D
  