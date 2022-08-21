def translate(text, lang_id="zu_Latn"):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./models/nllb_ru_kbd_44K")
    model = AutoModelForSeq2SeqLM.from_pretrained("./models/nllb_ru_kbd_44K")

    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_id], num_beams=4, num_return_sequences=4
    )

    for translation in tokenizer.batch_decode(translated_tokens, skip_special_tokens=True):
        print(translation)

# call translate from the command line
if __name__ == "__main__":
    import sys

    translate(sys.argv[1])