# Kitty Catalog AI

AI cat meme retrieval using image metadata. Given a photo album, this program can assign metadata to each photo or video as "tags", "captions", and "context". Then, to retrieve from the album, a user can query a custom context, and using cosine simularity, returns the appropriate cat meme with the appropriate context.

## WARNINGJ

Photo albums of cat videos are not provided in this repository for privacy reasons. You must supply your own photoalbum. However, we do supply our "catlib.db", which still has the metadata, just not the actual photos. We provide our catlib.db so our results can be remade.

You can use any photoalbum, not including cats. If you need one, here is are open source cat meme databases: 
small album, great classic cat memes: https://huggingface.co/datasets/pielet/cat-memes
large album, but filled with non meme cats: https://huggingface.co/datasets/Hack337/tg_cat_gifs/tree/main

## HOW TO RUN

1. Install the required dependencies with "pip install -r requirements.txt"

2. Run "ollama serve" in the terminal.

3. In the root of this project folder, store a folder titled "media", and store ALL photos/videos in it.

4. Run in the terminal "python create_db.py" to create catlib.db, where metadata will be stored.

5. Run in the terminal "python generate-metadata.py", it should take a long time to run. This will assign metadata to each photo.

6. (optional) Run in the terminal "streamlit run app.py", this will open a web page to allow you to see and edit metadata.

7. To test the model, you can run in the terminal "python tests/generate_tests.py" and then run "streamlit run evaluate_tests.py", you can manually score the selected cat memes. Alternatively, you have the option to try your own custom queries.

8. When ready for deployment, code is available to make a custom telegram bot, just supply the token in a new file "token.txt"