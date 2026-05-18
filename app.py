import os
import sqlite3
import streamlit as st
import shutil

DB_PATH = "catlib.db"


def get_connection():
    return sqlite3.connect(DB_PATH)

def autosave_current_item():
    update_media_item(
        st.session_state.current_id,
        st.session_state.get(f"caption_{st.session_state.current_id}", ""),
        st.session_state.get(f"tags_{st.session_state.current_id}", ""),
        st.session_state.get(f"vibe_{st.session_state.current_id}", ""),
        st.session_state.get(f"favorite_{st.session_state.current_id}", False)
    )


def get_media_item(item_id=None):
    conn = get_connection()
    cursor = conn.cursor()

    if item_id is None:
        cursor.execute("""
            SELECT id, file_path, file_name, media_type, caption, tags, vibe, favorite
            FROM media_items
            ORDER BY id
            LIMIT 1;
        """)
    else:
        cursor.execute("""
            SELECT id, file_path, file_name, media_type, caption, tags, vibe, favorite
            FROM media_items
            WHERE id = ?;
        """, (item_id,))

    row = cursor.fetchone()
    conn.close()
    return row


def get_next_id(current_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id
        FROM media_items
        WHERE id > ?
        ORDER BY id
        LIMIT 1;
    """, (current_id,))

    row = cursor.fetchone()
    conn.close()
    return row[0] if row else current_id


def get_previous_id(current_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id
        FROM media_items
        WHERE id < ?
        ORDER BY id DESC
        LIMIT 1;
    """, (current_id,))

    row = cursor.fetchone()
    conn.close()
    return row[0] if row else current_id


def get_next_empty_id(current_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id
        FROM media_items
        WHERE id > ?
          AND (
            caption = ''
            OR tags = ''
            OR vibe = ''
          )
        ORDER BY id
        LIMIT 1;
    """, (current_id,))

    row = cursor.fetchone()
    conn.close()
    return row[0] if row else current_id


def get_previous_empty_id(current_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id
        FROM media_items
        WHERE id < ?
          AND (
            caption = ''
            OR tags = ''
            OR vibe = ''
          )
        ORDER BY id DESC
        LIMIT 1;
    """, (current_id,))

    row = cursor.fetchone()
    conn.close()
    return row[0] if row else current_id


def get_progress_counts():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM media_items;")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM media_items
        WHERE caption != ''
          AND tags != ''
          AND vibe != '';
    """)
    complete = cursor.fetchone()[0]

    conn.close()
    return total, complete


def get_item_by_filename_search(search_text):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, file_name
        FROM media_items
        WHERE file_name LIKE ?
        ORDER BY id
        LIMIT 20;
    """, (f"%{search_text}%",))

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_item_by_metadata_search(search_text):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, file_name
        FROM media_items
        WHERE caption LIKE ?
           OR tags LIKE ?
           OR vibe LIKE ?
        ORDER BY id
        LIMIT 20;
    """, (
        f"%{search_text}%",
        f"%{search_text}%",
        f"%{search_text}%"
    ))

    rows = cursor.fetchall()
    conn.close()
    return rows


def media_id_exists(item_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id
        FROM media_items
        WHERE id = ?;
    """, (item_id,))

    row = cursor.fetchone()
    conn.close()
    return row is not None


def update_media_item(item_id, caption, tags, vibe, favorite):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE media_items
        SET caption = ?,
            tags = ?,
            vibe = ?,
            favorite = ?
        WHERE id = ?;
    """, (
        caption,
        tags,
        vibe,
        1 if favorite else 0,
        item_id
    ))

    conn.commit()
    conn.close()


def delete_media_item_and_file(item_id, file_path):
    deleted_folder = "deleted_media"
    os.makedirs(deleted_folder, exist_ok=True)

    file_name = os.path.basename(file_path)
    deleted_path = os.path.join(deleted_folder, file_name)

    base, ext = os.path.splitext(deleted_path)
    counter = 1

    while os.path.exists(deleted_path):
        deleted_path = f"{base}_{counter}{ext}"
        counter += 1

    if os.path.exists(file_path):
        shutil.move(file_path, deleted_path)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM media_items
        WHERE id = ?;
    """, (item_id,))

    conn.commit()
    conn.close()


def go_to_best_after_delete(current_id):
    next_id = get_next_id(current_id)

    if next_id != current_id:
        return next_id

    previous_id = get_previous_id(current_id)

    if previous_id != current_id:
        return previous_id

    return None


def go_previous():
    autosave_current_item()
    current_id = st.session_state.current_id
    previous_id = get_previous_id(current_id)

    if previous_id == current_id:
        st.toast("Beginning of list!")
    else:
        st.session_state.current_id = previous_id


def go_next():
    autosave_current_item()
    current_id = st.session_state.current_id
    next_id = get_next_id(current_id)

    if next_id == current_id:
        st.toast("End of list!")
    else:
        st.session_state.current_id = next_id


def go_previous_empty():
    autosave_current_item()
    current_id = st.session_state.current_id
    previous_id = get_previous_empty_id(current_id)

    if previous_id == current_id:
        st.toast("No previous empty items!")
    else:
        st.session_state.current_id = previous_id


def go_next_empty():
    autosave_current_item()
    current_id = st.session_state.current_id
    next_id = get_next_empty_id(current_id)

    if next_id == current_id:
        st.toast("No next empty items!")
    else:
        st.session_state.current_id = next_id


st.set_page_config(page_title="Kitty Catalog", layout="wide")
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}

div[data-testid="stTextArea"] textarea {
    min-height: 80px;
}

div[data-testid="stTextInput"] {
    margin-bottom: -0.5rem;
}

div[data-testid="stCheckbox"] {
    margin-top: -0.5rem;
    margin-bottom: -0.5rem;
}

hr {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Kitty Catalog")

total_items, completed_items = get_progress_counts()
remaining_items = total_items - completed_items

st.write(
    f"Progress: {completed_items} / {total_items} fully labeled "
    f"({remaining_items} remaining)"
)

if total_items > 0:
    st.progress(completed_items / total_items)

if "current_id" not in st.session_state:
    first_item = get_media_item()

    if first_item is None:
        st.error("No media found. Run scan_media.py first.")
        st.stop()

    st.session_state.current_id = first_item[0]

item = get_media_item(st.session_state.current_id)

if item is None:
    st.error("Media item not found.")
    st.stop()

item_id, file_path, file_name, media_type, caption, tags, vibe, favorite = item

left, right = st.columns([1, 2])

with left:
    st.subheader(file_name)

    st.write(f"ID: {item_id}")
    st.write(f"Type: {media_type}")


    if media_type == "image":
        st.image(file_path, use_column_width=True)

    elif media_type == "video":
        st.video(file_path, autoplay=True, loop=True, muted=False)

    else:
        st.warning(f"Unknown media type: {media_type}")

with right:
    
    st.divider()

    st.markdown("### Jump / Search")

    jump_id = st.number_input(
        "Jump to ID",
        min_value=1,
        step=1,
        key="jump_id"
    )

    if st.button("Go to ID"):
        if media_id_exists(jump_id):
            st.session_state.current_id = int(jump_id)
            st.rerun()
        else:
            st.warning("No media item found with that ID.")

    filename_search = st.text_input(
        "Search filename",
        key="filename_search"
    )

    if filename_search:
        matches = get_item_by_filename_search(filename_search)

        if matches:
            options = {
                f"{row[0]} - {row[1]}": row[0]
                for row in matches
            }

            selected = st.selectbox(
                "Filename matches",
                options=list(options.keys())
            )

            if st.button("Go to selected file"):
                st.session_state.current_id = options[selected]
                st.rerun()
        else:
            st.info("No filename matches found.")

    metadata_search = st.text_input(
        "Search caption / tags / vibe",
        key="metadata_search"
    )

    if metadata_search:
        matches = get_item_by_metadata_search(metadata_search)

        if matches:
            options = {
                f"{row[0]} - {row[1]}": row[0]
                for row in matches
            }

            selected = st.selectbox(
                "Metadata matches",
                options=list(options.keys())
            )

            if st.button("Go to selected metadata match"):
                st.session_state.current_id = options[selected]
                st.rerun()
        else:
            st.info("No metadata matches found.")

    st.divider()

    st.markdown("### Metadata")

    new_caption = st.text_area(
        "Caption",
        value=caption or "",
        key=f"caption_{item_id}",
        help="""
        Describe what is literally happening in the image/video.

        Example: 'cat sitting sadly in shopping cart'
        """
    )

    new_tags = st.text_area(
        "Tags",
        value=tags or "",
        key=f"tags_{item_id}",
        help="""
        List of comma-separated searchable concepts.

        Example: 'shopping cart, orange cat, crying, blue, thumbs up'
        """
    )

    new_vibe = st.text_input(
        "Vibe",
        value=vibe or "",
        key=f"vibe_{item_id}",
        help="""
        List of single words to describe emotional or social tone.

        Examples: chaotic, tragic, supportive, dramatic, resigned
        """
    )

    new_favorite = st.checkbox(
        "Favorite",
        value=bool(favorite),
        key=f"favorite_{item_id}"
    )

    if st.button("Save", key=f"save_{item_id}", use_container_width=True):
        update_media_item(
            item_id,
            new_caption,
            new_tags,
            new_vibe,
            new_favorite
        )
        st.success("Saved.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.button("Previous", on_click=go_previous, use_container_width=True)

    with col2:
        st.button("Next", on_click=go_next, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.button("Previous empty", on_click=go_previous_empty, use_container_width=True)

    with col4:
        st.button("Next empty", on_click=go_next_empty, use_container_width=True)

    st.divider()

    delete_confirm = st.checkbox(
        "Confirm delete from catalog",
        key=f"delete_confirm_{item_id}"
    )

    if st.button("Delete from catalog", key=f"delete_{item_id}", use_container_width=True):
        if delete_confirm:
            next_item_id = go_to_best_after_delete(item_id)
            delete_media_item_and_file(item_id, file_path)

            if next_item_id is None:
                st.success("Deleted. No media items left.")
                st.stop()

            st.session_state.current_id = next_item_id
            st.success("Deleted from catalog.")
            st.rerun()
        else:
            st.warning("Check the confirm box first.")

    