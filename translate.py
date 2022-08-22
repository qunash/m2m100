def translate(text, model_path = "anzorq/nllb_ru-kbd_44K_", src_lang="zul_Latn", tgt_lang="rus_Cyrl"):

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang=src_lang)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang], num_beams=4, num_return_sequences=4
    )

    for translation in tokenizer.batch_decode(translated_tokens, skip_special_tokens=True):
        print(translation)

# call translate from the command line
if __name__ == "__main__":
    import sys

    translate(sys.argv[1])