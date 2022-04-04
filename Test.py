from transformers import EncoderDecoderModel, BertConfig, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer

if __name__ == '__main__':
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('checkpoint-step-010000', 'checkpoint-step-010000')
    print('Load Completed')
