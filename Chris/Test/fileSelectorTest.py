
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display

folder_box = widgets.Text(
    description='Folder:',
    placeholder='Enter folder path (e.g., C:\\data or /home/you/data)'
)
browse_hint = widgets.HTML("<i>Tip: paste a path from VS Code Explorer or terminal.</i>")
run_btn = widgets.Button(description='Use folder', disabled=True)
status = widgets.HTML()

def on_change(change):
    p = Path(folder_box.value).expanduser()
    if p.is_dir():
        run_btn.disabled = False
        status.value = f"<b>Found:</b> {p}"
    else:
        run_btn.disabled = True
        status.value = "<b>Not a folder:</b> please enter a valid path."

folder_box.observe(on_change, names='value')

def on_run_clicked(b):
    p = Path(folder_box.value).expanduser()
    print(f"Selected folder: {p}")
    items = list(p.iterdir())
    print(f"Found {len(items)} items; showing first 10:")
    for itm in items[:10]:
        print(" -", itm.name)

run_btn.on_click(on_run_clicked)
display(widgets.VBox([browse_hint, folder_box, status, run_btn]))
