"""
Usage: python scripts/annotate_images.py

Description:
    Launches a GUI for annotating images with bounding boxes for YOLOv12 training.
    
Parameters:
    None. Configuration is handled via global variables at the top of the script.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import math
import os
import glob
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE = r"V:\PollinatorINaturalistData\extractedImagesIreenTest"
PATH_TO_CLASSES_DEFINITION_FILE = os.path.join(DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE, "classes.txt")

# ──────────────────────────────────────────────────────────────────────
#  AI-assisted suggestions  (Grounding DINO detection pipeline)
# ──────────────────────────────────────────────────────────────────────
USE_AI_SUGGESTIONS = True
# *tiny* for speed, *base* for accuracy.  Tiny is ~3x faster.
MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
# Enforce GPU usage (fails fast if CUDA is not available).
REQUIRE_CUDA = True
CUDA_DEVICE_INDEX = 0
# FP16 works with grounding-dino-tiny, halves VRAM & doubles speed.
# NOTE: Some transformers versions have float/Half bias mismatches.
# Set to False if you see c10::Half dtype errors.
USE_FP16 = False
# Primary detection threshold – higher = fewer, more confident boxes.
BOX_THRESHOLD = 0.35
# Text-matching threshold – higher = only strong text-to-region matches.
# Prevents body parts / background regions from loosely matching prompts.
TEXT_THRESHOLD = 0.25
# Minimum box edge in pixels (original image coords).
MIN_BOX_SIZE_PIXELS = 5
# Inference resolution (longest side). 800 is the DINO training default.
MAX_INFERENCE_SIZE = 800
# Maximum predicted boxes after NMS (rarely > 3 insects in one image).
MAX_BOXES_PER_IMAGE = 5
# Drop boxes covering more than this fraction of the image.
MAX_BOX_AREA_RATIO = 0.60
# Prefetch predictions for upcoming images.
PREFETCH_AHEAD = 20
# Max concurrent prefetch workers (GPU is the bottleneck, so keep low).
PREFETCH_WORKERS = 4

# ── Negative labels ──────────────────────────────────────────────────
# These labels are passed alongside positive prompts in a SINGLE
# inference call (zero extra cost).  Any detection matching a negative
# label is discarded.  This teaches the model "that's a flower, not a bee".
NEGATIVE_LABELS = [
    "flower", "petal", "leaf", "rock", "stone", "branch",
    "stem", "bud", "grass", "twig", "bark",
]

# ── Hard NMS ──────────────────────────────────────────────────────────
# Strict IoU-based NMS: aggressively merges overlapping boxes so that
# each insect gets exactly ONE bounding box.  IoU 0.3 means any box
# overlapping 30%+ with a higher-scoring box is removed.
NMS_IOU_THRESHOLD = 0.3

# ── Multi-scale ensemble ─────────────────────────────────────────────
# Run detection at several scales and merge.  Catches insects at
# different apparent sizes.  Disabled by default for speed.
USE_MULTI_SCALE = False
MULTI_SCALE_FACTORS = [0.75, 1.0, 1.3]

# ── Test-time augmentation ───────────────────────────────────────────
# Horizontal flip TTA doubles inference time but can improve recall.
USE_HORIZONTAL_FLIP_TTA = False

# ──────────────────────────────────────────────────────────────────────
#  Species-aware prompt system for Grounding DINO
# ──────────────────────────────────────────────────────────────────────
# Grounding DINO works with SHORT NOUN PHRASES, not sentences.
# "honey bee" >> "a photo of apis mellifera"
# Multiple labels run in parallel and get merged via Soft-NMS.
#
# Each species maps to 3 prompts:
#   [specific common name, group name, visual descriptor]
# ──────────────────────────────────────────────────────────────────────
SPECIES_PROMPTS: Dict[str, List[str]] = {
    # Butterflies
    "aglais_urticae":           ["small tortoiseshell butterfly", "butterfly", "orange and black butterfly"],
    "pieris_brassicae":         ["large white butterfly", "white butterfly", "butterfly"],
    "pieris_rapae":             ["small white butterfly", "white butterfly", "butterfly"],
    # Honey bees
    "apis_mellifera":           ["honey bee", "honeybee", "bee"],
    # Bumblebees
    "bombus_lapidarius":        ["red-tailed bumblebee", "bumblebee", "black bee with red tail"],
    "bombus_pascuorum":         ["common carder bumblebee", "bumblebee", "brown furry bee"],
    "bombus_sylvarum":          ["shrill carder bumblebee", "bumblebee", "grey fuzzy bee"],
    "bombus_terrestris":        ["buff-tailed bumblebee", "bumblebee", "large fuzzy bee"],
    # Hoverflies (Syrphidae)
    "chrysogaster_solstitialis":["hoverfly", "small hoverfly", "tiny fly on flower"],
    "eristalis_arbustorum":     ["hoverfly", "drone fly", "bee-mimicking fly"],
    "eristalis_intricaria":     ["hoverfly", "furry drone fly", "hairy bee-like fly"],
    "eristalis_pertinax":       ["hoverfly", "drone fly", "bee-mimicking fly"],
    "eristalis_similis":        ["hoverfly", "drone fly", "bee-mimicking fly"],
    "eristalis_tenax":          ["hoverfly", "drone fly", "bee-mimicking fly"],
    "helophilus_hybridus":      ["hoverfly", "striped hoverfly", "yellow and black fly"],
    "helophilus_pendulus":      ["hoverfly", "striped hoverfly", "yellow and black fly"],
    "helophilus_trivittatus":   ["hoverfly", "large striped hoverfly", "yellow and black fly"],
    "melanostoma_mellinum":     ["hoverfly", "small hoverfly", "slender fly on flower"],
    "myathropa_florea":         ["hoverfly", "dead head hoverfly", "fly with skull marking"],
    "pyrophaena_granditarsis":  ["hoverfly", "small hoverfly", "tiny fly on flower"],
    "sericomyia_silentis":      ["hoverfly", "bog hoverfly", "wasp-mimicking fly"],
    "sphaerophoria_scripta":    ["hoverfly", "long hoverfly", "slender yellow-banded fly"],
    "syritta_pipiens":          ["hoverfly", "thick-legged hoverfly", "small dark fly"],
    # Beetles
    "anthaxia_species":         ["beetle", "jewel beetle", "small metallic beetle"],
    "cryptocephalus_species":   ["beetle", "leaf beetle", "small round beetle"],
}

# Genus-level fallback for species not in the dictionary above.
GENUS_TO_COMMON: Dict[str, Tuple[str, str]] = {
    # Bees
    "apis":           ("bee",       "honey bee"),
    "bombus":         ("bumblebee", "fuzzy bumblebee"),
    "andrena":        ("bee",       "mining bee"),
    "osmia":          ("bee",       "mason bee"),
    "xylocopa":       ("bee",       "carpenter bee"),
    "anthophora":     ("bee",       "flower bee"),
    "colletes":       ("bee",       "plasterer bee"),
    "megachile":      ("bee",       "leafcutter bee"),
    "nomada":         ("bee",       "cuckoo bee"),
    "lasioglossum":   ("bee",       "sweat bee"),
    "halictus":       ("bee",       "sweat bee"),
    # Butterflies
    "pieris":         ("butterfly", "white butterfly"),
    "aglais":         ("butterfly", "tortoiseshell butterfly"),
    "vanessa":        ("butterfly", "painted lady butterfly"),
    "polygonia":      ("butterfly", "comma butterfly"),
    "lycaena":        ("butterfly", "copper butterfly"),
    "maniola":        ("butterfly", "meadow brown butterfly"),
    # Hoverflies
    "eristalis":      ("hoverfly",  "drone fly"),
    "helophilus":     ("hoverfly",  "striped hoverfly"),
    "melanostoma":    ("hoverfly",  "small hoverfly"),
    "myathropa":      ("hoverfly",  "dead head hoverfly"),
    "chrysogaster":   ("hoverfly",  "small hoverfly"),
    "pyrophaena":     ("hoverfly",  "small hoverfly"),
    "syrphus":        ("hoverfly",  "common hoverfly"),
    "episyrphus":     ("hoverfly",  "marmalade hoverfly"),
    "volucella":      ("hoverfly",  "large hoverfly"),
    "sphaerophoria":  ("hoverfly",  "long hoverfly"),
    "sericomyia":     ("hoverfly",  "bog hoverfly"),
    "syritta":        ("hoverfly",  "thick-legged hoverfly"),
    "rhingia":        ("hoverfly",  "snout hoverfly"),
    "xylota":         ("hoverfly",  "wood hoverfly"),
    "cheilosia":      ("hoverfly",  "cheilosia hoverfly"),
    "platycheirus":   ("hoverfly",  "platycheirus hoverfly"),
    "baccha":         ("hoverfly",  "slender hoverfly"),
    # Wasps
    "vespa":          ("wasp",      "hornet"),
    "vespula":        ("wasp",      "yellowjacket wasp"),
    "polistes":       ("wasp",      "paper wasp"),
    # Beetles
    "anthaxia":       ("beetle",    "jewel beetle"),
    "cryptocephalus": ("beetle",    "leaf beetle"),
    "cetonia":        ("beetle",    "rose chafer beetle"),
    "coccinella":     ("ladybird",  "ladybug"),
    "oxythyrea":      ("beetle",    "white-spotted beetle"),
    "trichius":       ("beetle",    "bee beetle"),
}

# Ultimate fallback when neither species nor genus is recognised.
GENERIC_PROMPTS = ["insect", "bee", "butterfly", "fly", "hoverfly", "beetle"]


def build_detection_prompts(folder_name: str) -> List[str]:
    """Build optimal Grounding DINO prompts for a species folder.

    Priority: exact species match → genus-level inference → generic fallback.
    Returns 3–6 short noun-phrase prompts (no sentence templates).
    """
    # 1. Exact species match
    if folder_name in SPECIES_PROMPTS:
        return SPECIES_PROMPTS[folder_name]

    # 2. Genus-level inference  (first part of folder name)
    genus = folder_name.split("_")[0].lower()
    if genus in GENUS_TO_COMMON:
        group_name, common_name = GENUS_TO_COMMON[genus]
        readable = folder_name.replace("_", " ")
        return [common_name, group_name, f"{group_name} on flower"]

    # 3. Generic fallback
    return GENERIC_PROMPTS

class InsectImageAnnotationTool:
    def __init__(self, main_tkinter_window):
        self.main_application_window = main_tkinter_window
        self.main_application_window.title("YOLOv12 Insect Annotation Tool")
        self.main_application_window.geometry("1200x900")
        
        self.dictionary_mapping_class_names_to_ids = {}
        self.dictionary_mapping_ids_to_class_names = {}
        self.load_and_synchronize_classes_from_directories()

        self.ai_model = None
        self.ai_processor = None
        self.ai_device = "cpu"
        self._gpu_lock = threading.Lock()
        if USE_AI_SUGGESTIONS:
            self.initialize_ai_detector()
        
        self.name_of_currently_selected_folder = None
        self.list_of_pending_image_file_paths = []
        self.index_of_current_image_in_list = 0
        self.full_path_to_current_image_file = None
        self.current_image_object_for_tkinter_display = None
        self.original_image_object_from_pillow = None
        
        self.list_of_drawn_rectangles_data = []
        self.coordinate_x_where_mouse_press_started = None
        self.coordinate_y_where_mouse_press_started = None
        self.identifier_of_rectangle_currently_being_drawn = None
        self.boolean_flag_indicating_drawing_in_progress = False
        self.scaling_ratio_applied_to_displayed_image = 1.0
        self.boolean_flag_ai_suggestions_active = False
        self.list_of_ai_rectangles_ids = []
        self.boolean_flag_deletion_selection_in_progress = False
        self.identifier_of_deletion_selection_rectangle = None
        self.deletion_selection_start = None
        self.loading_progress_bar = None
        self.current_load_token = 0
        self.prefetch_cache = {}
        self.prefetch_in_progress = set()
        
        self.main_ui_container_frame = tk.Frame(self.main_application_window)
        self.main_ui_container_frame.pack(fill=tk.BOTH, expand=True)
        
        self.display_folder_selection_menu_screen()

    def initialize_ai_detector(self):
        try:
            # Properly initialize CUDA before checking availability
            cuda_available = False
            if torch.cuda.is_available():
                try:
                    # Force CUDA initialization by creating a small tensor
                    _ = torch.zeros(1, device=f"cuda:{CUDA_DEVICE_INDEX}")
                    cuda_available = True
                    print(f"[AI] CUDA initialized successfully on device {CUDA_DEVICE_INDEX}")
                    print(f"[AI] GPU: {torch.cuda.get_device_name(CUDA_DEVICE_INDEX)}")
                except Exception as cuda_error:
                    print(f"[AI] CUDA initialization failed: {cuda_error}")
                    cuda_available = False
            
            if REQUIRE_CUDA and not cuda_available:
                messagebox.showerror(
                    "CUDA required",
                    "CUDA is not available or failed to initialize.\n"
                    "Install a CUDA-enabled PyTorch build and NVIDIA drivers.\n\n"
                    "To run without GPU, set REQUIRE_CUDA = False",
                )
                raise RuntimeError("CUDA required but not available")

            if cuda_available:
                torch.backends.cudnn.benchmark = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

            self.ai_device = f"cuda:{CUDA_DEVICE_INDEX}" if cuda_available else "cpu"
            dtype = torch.float16 if (cuda_available and USE_FP16) else torch.float32

            print(f"[AI] Loading model: {MODEL_NAME}")
            print(f"[AI] Device: {self.ai_device}  dtype: {dtype}")

            # ── Direct model loading (no pipeline wrapper) ───────────
            self.ai_processor = AutoProcessor.from_pretrained(MODEL_NAME)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                MODEL_NAME, torch_dtype=dtype,
            )
            model = model.to(self.ai_device)
            model.eval()

            # ── torch.compile: fuse GPU ops for speedup ─────────────────
            # On Windows, Triton/inductor is not available, so we use
            # "eager" mode which still benefits from graph capture.
            try:
                import sys
                if sys.platform == "win32":
                    # Skip torch.compile on Windows — no Triton support
                    print("[AI] torch.compile() skipped (Windows — no Triton)")
                else:
                    model = torch.compile(model, mode="reduce-overhead")
                    print("[AI] torch.compile() enabled (reduce-overhead)")
            except Exception as compile_err:
                print(f"[AI] torch.compile() not available: {compile_err}")

            self.ai_model = model

            # Warm up with a dummy forward pass to trigger compilation
            try:
                dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
                inputs = self.ai_processor(
                    images=dummy, text="insect.", return_tensors="pt"
                ).to(self.ai_device)
                with torch.inference_mode():
                    _ = self.ai_model(**inputs)
                del inputs, dummy
                torch.cuda.empty_cache()
                print("[AI] Warm-up forward pass complete")
            except Exception:
                pass

            print("[AI] Model loaded successfully")
        except Exception as error_message:
            print(f"[AI] Model failed to initialize: {error_message}")
            import traceback
            traceback.print_exc()
            self.ai_model = None

    def load_and_synchronize_classes_from_directories(self):
        if not os.path.exists(DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE):
            os.makedirs(DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE)
            
        list_of_subdirectories = [
            directory_name for directory_name in os.listdir(DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE) 
            if os.path.isdir(os.path.join(DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE, directory_name))
        ]
        list_of_subdirectories.sort()
        
        list_of_existing_classes = []
        if os.path.exists(PATH_TO_CLASSES_DEFINITION_FILE):
            with open(PATH_TO_CLASSES_DEFINITION_FILE, 'r') as file_handle:
                list_of_existing_classes = [line.strip() for line in file_handle.readlines() if line.strip()]
        
        boolean_flag_classes_have_changed = False
        for directory_name in list_of_subdirectories:
            if directory_name not in list_of_existing_classes:
                list_of_existing_classes.append(directory_name)
                boolean_flag_classes_have_changed = True
        
        if boolean_flag_classes_have_changed or not os.path.exists(PATH_TO_CLASSES_DEFINITION_FILE):
            with open(PATH_TO_CLASSES_DEFINITION_FILE, 'w') as file_handle:
                for class_name in list_of_existing_classes:
                    file_handle.write(f"{class_name}\n")
        
        self.dictionary_mapping_class_names_to_ids = {name: index for index, name in enumerate(list_of_existing_classes)}
        self.dictionary_mapping_ids_to_class_names = {index: name for index, name in enumerate(list_of_existing_classes)}
        print(f"Classes loaded: {len(self.dictionary_mapping_class_names_to_ids)}")

    def remove_all_widgets_from_main_container(self):
        for widget in self.main_ui_container_frame.winfo_children():
            widget.destroy()

    def display_folder_selection_menu_screen(self):
        self.remove_all_widgets_from_main_container()
        self.main_application_window.unbind('<y>')
        self.main_application_window.unbind('<n>')
        
        header_label_widget = tk.Label(self.main_ui_container_frame, text="Select Species Folder", font=("Arial", 16, "bold"))
        header_label_widget.pack(pady=20)
        
        canvas_for_scrolling = tk.Canvas(self.main_ui_container_frame)
        scrollbar_widget = ttk.Scrollbar(self.main_ui_container_frame, orient="vertical", command=canvas_for_scrolling.yview)
        frame_inside_canvas = ttk.Frame(canvas_for_scrolling)

        frame_inside_canvas.bind(
            "<Configure>",
            lambda event: canvas_for_scrolling.configure(scrollregion=canvas_for_scrolling.bbox("all"))
        )

        canvas_for_scrolling.create_window((0, 0), window=frame_inside_canvas, anchor="nw")
        canvas_for_scrolling.configure(yscrollcommand=scrollbar_widget.set)

        canvas_for_scrolling.pack(side="left", fill="both", expand=True, padx=20)
        scrollbar_widget.pack(side="right", fill="y")

        list_of_subdirectories = [
            directory_name for directory_name in os.listdir(DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE) 
            if os.path.isdir(os.path.join(DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE, directory_name))
        ]
        list_of_subdirectories.sort()
        
        for folder_name in list_of_subdirectories:
            full_path_to_folder = os.path.join(DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE, folder_name)
            list_of_all_images_in_folder = glob.glob(os.path.join(full_path_to_folder, "*.jpg"))
            total_count_of_images = len(list_of_all_images_in_folder)
            
            count_of_processed_images = 0
            for image_path in list_of_all_images_in_folder:
                path_to_annotation_text_file = os.path.splitext(image_path)[0] + ".txt"
                if os.path.exists(path_to_annotation_text_file):
                    count_of_processed_images += 1
            
            count_of_remaining_images = total_count_of_images - count_of_processed_images
            
            button_text_label = f"{folder_name} (Remaining: {count_of_remaining_images} / Total: {total_count_of_images})"
            button_state = tk.NORMAL if count_of_remaining_images > 0 else tk.DISABLED
            
            folder_selection_button = tk.Button(
                frame_inside_canvas, 
                text=button_text_label, 
                font=("Arial", 12), 
                width=60, 
                height=2,
                state=button_state,
                command=lambda selected_folder=folder_name: self.initialize_annotation_session_for_folder(selected_folder)
            )
            folder_selection_button.pack(pady=5)

    def initialize_annotation_session_for_folder(self, folder_name):
        self.name_of_currently_selected_folder = folder_name
        full_path_to_folder = os.path.join(DIRECTORY_CONTAINING_IMAGES_TO_ANNOTATE, folder_name)
        
        list_of_all_images_in_folder = glob.glob(os.path.join(full_path_to_folder, "*.jpg"))
        self.list_of_pending_image_file_paths = []
        for image_path in list_of_all_images_in_folder:
            path_to_annotation_text_file = os.path.splitext(image_path)[0] + ".txt"
            if not os.path.exists(path_to_annotation_text_file):
                self.list_of_pending_image_file_paths.append(image_path)
        
        self.list_of_pending_image_file_paths.sort()
        self.index_of_current_image_in_list = 0
        
        if not self.list_of_pending_image_file_paths:
            messagebox.showinfo("Info", "No images remaining in this folder!")
            self.display_folder_selection_menu_screen()
            return
            
        self.configure_annotation_interface_widgets()
        self.load_and_display_current_image_file()

    def configure_annotation_interface_widgets(self):
        self.remove_all_widgets_from_main_container()
        
        top_navigation_bar_frame = tk.Frame(self.main_ui_container_frame)
        top_navigation_bar_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(top_navigation_bar_frame, text="Back to Menu", command=self.display_folder_selection_menu_screen).pack(side=tk.LEFT)
        self.label_widget_for_image_info = tk.Label(top_navigation_bar_frame, text="", font=("Arial", 12, "bold"))
        self.label_widget_for_image_info.pack(side=tk.LEFT, padx=20)

        self.loading_progress_bar = ttk.Progressbar(top_navigation_bar_frame, mode="indeterminate", length=160)
        self.loading_progress_bar.pack(side=tk.LEFT, padx=10)
        self.loading_progress_bar.stop()
        self.loading_progress_bar.pack_forget()
        
        tk.Label(top_navigation_bar_frame, text="Controls: Left-drag add | Right-drag remove | 'y': Save | 'n': Delete").pack(side=tk.RIGHT)
        
        self.canvas_widget_for_drawing = tk.Canvas(self.main_ui_container_frame, cursor="cross")
        self.canvas_widget_for_drawing.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas_widget_for_drawing.bind("<ButtonPress-1>", self.handle_mouse_button_press_event)
        self.canvas_widget_for_drawing.bind("<B1-Motion>", self.handle_mouse_drag_event)
        self.canvas_widget_for_drawing.bind("<ButtonRelease-1>", self.handle_mouse_button_release_event)
        self.canvas_widget_for_drawing.bind("<ButtonPress-3>", self.handle_right_button_press_event)
        self.canvas_widget_for_drawing.bind("<B3-Motion>", self.handle_right_button_drag_event)
        self.canvas_widget_for_drawing.bind("<ButtonRelease-3>", self.handle_right_button_release_event)
        
        self.main_application_window.bind('<y>', lambda event: self.save_annotations_to_file_and_proceed_to_next_image())
        self.main_application_window.bind('<n>', lambda event: self.delete_current_image_file_and_proceed_to_next_image())

    def load_and_display_current_image_file(self):
        if self.index_of_current_image_in_list >= len(self.list_of_pending_image_file_paths):
            messagebox.showinfo("Done", "Folder completed!")
            self.display_folder_selection_menu_screen()
            return
            
        self.full_path_to_current_image_file = self.list_of_pending_image_file_paths[self.index_of_current_image_in_list]
        self.list_of_drawn_rectangles_data = []
        self.list_of_ai_rectangles_ids = []
        self.boolean_flag_ai_suggestions_active = False
        self.canvas_widget_for_drawing.delete("all")
        self.current_load_token += 1
        current_token = self.current_load_token
        self.start_loading_indicator()
        
        count_of_remaining_images = len(self.list_of_pending_image_file_paths) - self.index_of_current_image_in_list
        self.label_widget_for_image_info.config(text=f"Folder: {self.name_of_currently_selected_folder} | Image: {os.path.basename(self.full_path_to_current_image_file)} | Remaining: {count_of_remaining_images}")
        
        try:
            self.original_image_object_from_pillow = Image.open(self.full_path_to_current_image_file)
            
            width_of_screen_area = self.main_application_window.winfo_width() - 50
            height_of_screen_area = self.main_application_window.winfo_height() - 100
            
            original_image_width, original_image_height = self.original_image_object_from_pillow.size
            self.scaling_ratio_applied_to_displayed_image = 1.0
            
            minimum_width_for_comfortable_viewing = 800
            minimum_height_for_comfortable_viewing = 600
            
            calculated_scaling_ratio = 1.0
            
            if original_image_width > width_of_screen_area or original_image_height > height_of_screen_area:
                ratio_width = width_of_screen_area / original_image_width
                ratio_height = height_of_screen_area / original_image_height
                calculated_scaling_ratio = min(ratio_width, ratio_height)
            
            elif original_image_width < minimum_width_for_comfortable_viewing or original_image_height < minimum_height_for_comfortable_viewing:
                ratio_width = minimum_width_for_comfortable_viewing / original_image_width
                ratio_height = minimum_height_for_comfortable_viewing / original_image_height
                calculated_scaling_ratio = min(ratio_width, ratio_height)
                
                potential_new_width = original_image_width * calculated_scaling_ratio
                potential_new_height = original_image_height * calculated_scaling_ratio
                
                if potential_new_width > width_of_screen_area or potential_new_height > height_of_screen_area:
                    ratio_width = width_of_screen_area / original_image_width
                    ratio_height = height_of_screen_area / original_image_height
                    calculated_scaling_ratio = min(ratio_width, ratio_height)

            self.scaling_ratio_applied_to_displayed_image = calculated_scaling_ratio
            
            if self.scaling_ratio_applied_to_displayed_image != 1.0:
                new_width = int(original_image_width * self.scaling_ratio_applied_to_displayed_image)
                new_height = int(original_image_height * self.scaling_ratio_applied_to_displayed_image)
                image_to_display = self.original_image_object_from_pillow.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                image_to_display = self.original_image_object_from_pillow
                
            self.current_image_object_for_tkinter_display = ImageTk.PhotoImage(image_to_display)
            self.canvas_widget_for_drawing.create_image(0, 0, anchor=tk.NW, image=self.current_image_object_for_tkinter_display)
            self.canvas_widget_for_drawing.config(scrollregion=self.canvas_widget_for_drawing.bbox(tk.ALL))

            if USE_AI_SUGGESTIONS and self.ai_model is not None:
                self.populate_ai_suggestions_async(current_token)
            else:
                self.stop_loading_indicator()
            
        except Exception as error_message:
            print(f"Error loading image: {error_message}")
            self.stop_loading_indicator()
            self.delete_current_image_file_and_proceed_to_next_image()

    def start_loading_indicator(self):
        if self.loading_progress_bar is not None:
            self.loading_progress_bar.pack(side=tk.LEFT, padx=10)
            self.loading_progress_bar.start(10)

    def stop_loading_indicator(self):
        if self.loading_progress_bar is not None:
            self.loading_progress_bar.stop()
            self.loading_progress_bar.pack_forget()

    def populate_ai_suggestions_async(self, load_token):
        if self.original_image_object_from_pillow is None:
            self.stop_loading_indicator()
            return

        cached_boxes = self.prefetch_cache.pop(self.full_path_to_current_image_file, None)
        if cached_boxes is not None:
            self.apply_ai_suggestions(load_token, cached_boxes)
            self.start_prefetch_for_next_image()
            return

        def worker():
            kept_boxes = self.compute_ai_suggestions_for_image(self.original_image_object_from_pillow)
            self.main_application_window.after(
                0,
                lambda: self.apply_ai_suggestions(load_token, kept_boxes),
            )
            self.start_prefetch_for_next_image()

        threading.Thread(target=worker, daemon=True).start()

    def apply_ai_suggestions(self, load_token, kept_boxes):
        if load_token != self.current_load_token:
            return

        if not kept_boxes:
            self.stop_loading_indicator()
            return

        for (x1, y1, x2, y2, _) in kept_boxes:
            x1_scaled = x1 * self.scaling_ratio_applied_to_displayed_image
            y1_scaled = y1 * self.scaling_ratio_applied_to_displayed_image
            x2_scaled = x2 * self.scaling_ratio_applied_to_displayed_image
            y2_scaled = y2 * self.scaling_ratio_applied_to_displayed_image

            rectangle_id = self.canvas_widget_for_drawing.create_rectangle(
                x1_scaled, y1_scaled, x2_scaled, y2_scaled, outline="yellow", width=2, dash=(4, 2)
            )
            self.list_of_ai_rectangles_ids.append(rectangle_id)
            self.list_of_drawn_rectangles_data.append((x1_scaled, y1_scaled, x2_scaled, y2_scaled, rectangle_id))

        if self.list_of_ai_rectangles_ids:
            self.boolean_flag_ai_suggestions_active = True

        self.stop_loading_indicator()

    def start_prefetch_for_next_image(self):
        """Queue up to PREFETCH_AHEAD images for background prediction.
        Uses a shared ThreadPoolExecutor so we don't flood the GPU."""
        if not hasattr(self, '_prefetch_pool'):
            self._prefetch_pool = ThreadPoolExecutor(
                max_workers=PREFETCH_WORKERS,
                thread_name_prefix="prefetch",
            )

        for offset in range(1, PREFETCH_AHEAD + 1):
            next_index = self.index_of_current_image_in_list + offset
            if next_index >= len(self.list_of_pending_image_file_paths):
                break

            next_path = self.list_of_pending_image_file_paths[next_index]
            if next_path in self.prefetch_cache or next_path in self.prefetch_in_progress:
                continue

            self.prefetch_in_progress.add(next_path)

            def worker(path=next_path):
                try:
                    with Image.open(path) as next_image:
                        next_image.load()  # force read before closing file
                        boxes = self.compute_ai_suggestions_for_image(next_image)
                except Exception:
                    boxes = []

                def finalize():
                    self.prefetch_in_progress.discard(path)
                    self.prefetch_cache[path] = boxes

                self.main_application_window.after(0, finalize)

            self._prefetch_pool.submit(worker)

    # ── Core detection pipeline ────────────────────────────────────────

    def compute_ai_suggestions_for_image(self, image_object):
        """Run Grounding DINO with species-aware multi-prompt, multi-scale
        ensemble and Soft-NMS.  Returns list of (x1, y1, x2, y2, score)
        in *original* image coordinates."""
        if image_object is None:
            return []

        width, height = image_object.size
        image_rgb = image_object.convert("RGB")

        # Species-aware prompts  (NO template – raw noun phrases)
        prompts = build_detection_prompts(self.name_of_currently_selected_folder)

        all_boxes: List[Tuple[float, float, float, float, float]] = []

        # ── Multi-scale inference ────────────────────────────────────
        scales = MULTI_SCALE_FACTORS if USE_MULTI_SCALE else [1.0]

        for scale_factor in scales:
            target_size = int(MAX_INFERENCE_SIZE * scale_factor)
            longest_side = max(width, height)
            inference_scale = target_size / float(longest_side)

            if abs(inference_scale - 1.0) > 0.02:
                new_w = max(1, int(width * inference_scale))
                new_h = max(1, int(height * inference_scale))
                image_for_inference = image_rgb.resize(
                    (new_w, new_h), Image.Resampling.BILINEAR
                )
            else:
                image_for_inference = image_rgb
                inference_scale = 1.0

            # Run with species-specific prompts
            raw = self._run_inference(image_for_inference, prompts, BOX_THRESHOLD)
            self._rescale_and_collect(
                raw, inference_scale, width, height, all_boxes
            )

            # ── Horizontal-flip TTA ──────────────────────────────────
            if USE_HORIZONTAL_FLIP_TTA:
                flipped = image_for_inference.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )
                flip_w = image_for_inference.width
                raw_flip = self._run_inference(flipped, prompts, BOX_THRESHOLD)
                # Mirror x-coordinates back
                mirrored = [
                    (flip_w - x2, y1, flip_w - x1, y2, s)
                    for x1, y1, x2, y2, s in raw_flip
                ]
                self._rescale_and_collect(
                    mirrored, inference_scale, width, height, all_boxes
                )

        # ── Fallback with generic prompts if nothing found ──────────
        if not all_boxes:
            longest_side = max(width, height)
            inference_scale = MAX_INFERENCE_SIZE / float(longest_side)
            if abs(inference_scale - 1.0) > 0.02:
                new_w = max(1, int(width * inference_scale))
                new_h = max(1, int(height * inference_scale))
                fb_image = image_rgb.resize(
                    (new_w, new_h), Image.Resampling.BILINEAR
                )
            else:
                fb_image = image_rgb
                inference_scale = 1.0

            raw_fb = self._run_inference(fb_image, GENERIC_PROMPTS, BOX_THRESHOLD)
            self._rescale_and_collect(
                raw_fb, inference_scale, width, height, all_boxes
            )

        # ── Hard NMS ─────────────────────────────────────────────────
        kept = self.hard_nms(all_boxes, NMS_IOU_THRESHOLD, MAX_BOXES_PER_IMAGE)
        return kept

    # ── Inference helper ─────────────────────────────────────────────

    def _run_inference(
        self,
        image,
        prompts: List[str],
        threshold: float,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Run Grounding DINO directly (no pipeline wrapper).

        Positive + negative labels are combined into a single text prompt,
        separated by '. ' (Grounding DINO convention).  Any detection
        matching a negative label is discarded.  A GPU lock serialises
        access so prefetch threads don't fight over the device."""
        if self.ai_model is None or self.ai_processor is None:
            return []

        negative_set: Set[str] = set(NEGATIVE_LABELS)
        all_labels = list(prompts) + [l for l in NEGATIVE_LABELS if l not in prompts]

        # Grounding DINO expects labels joined with ". "
        text_prompt = ". ".join(all_labels) + "."

        w, h = image.size

        # CPU-side: tokenize + build pixel tensors (runs outside lock)
        inputs = self.ai_processor(
            images=image, text=text_prompt, return_tensors="pt"
        )

        try:
            # GPU-side: move tensors and run model under lock
            with self._gpu_lock:
                inputs = inputs.to(self.ai_device)
                with torch.inference_mode():
                    outputs = self.ai_model(**inputs)

                results = self.ai_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=threshold,
                    text_threshold=TEXT_THRESHOLD,
                    target_sizes=[(h, w)],
                )
        except Exception as err:
            print(f"[AI] inference error: {err}")
            return []

        boxes: List[Tuple[float, float, float, float, float]] = []
        if results:
            r = results[0]
            pred_boxes = r["boxes"]   # tensor (N, 4)  [x1,y1,x2,y2]
            pred_scores = r["scores"] # tensor (N,)
            pred_labels = r.get("labels", r.get("text", []))  # list[str]

            for i in range(len(pred_scores)):
                label = pred_labels[i] if i < len(pred_labels) else ""
                # Discard negative-label matches
                if label.strip().lower() in negative_set:
                    continue

                score = float(pred_scores[i])
                box = pred_boxes[i]
                boxes.append((
                    float(box[0]), float(box[1]),
                    float(box[2]), float(box[3]),
                    score,
                ))
        return boxes

    # ── Rescale + filter helper ──────────────────────────────────────

    @staticmethod
    def _rescale_and_collect(
        boxes: List[Tuple[float, float, float, float, float]],
        inference_scale: float,
        img_w: int,
        img_h: int,
        out: List[Tuple[float, float, float, float, float]],
    ) -> None:
        """Scale boxes back to original coords, clamp, filter."""
        for x1, y1, x2, y2, score in boxes:
            if inference_scale != 1.0:
                x1 /= inference_scale
                y1 /= inference_scale
                x2 /= inference_scale
                y2 /= inference_scale

            # Clamp to image bounds
            x1 = max(0.0, min(float(img_w), x1))
            y1 = max(0.0, min(float(img_h), y1))
            x2 = max(0.0, min(float(img_w), x2))
            y2 = max(0.0, min(float(img_h), y2))

            bw, bh = x2 - x1, y2 - y1
            if bw < MIN_BOX_SIZE_PIXELS or bh < MIN_BOX_SIZE_PIXELS:
                continue

            box_area = bw * bh
            img_area = max(1.0, img_w * img_h)
            if (box_area / img_area) >= MAX_BOX_AREA_RATIO:
                continue

            out.append((x1, y1, x2, y2, score))

    # ── IoU ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[2], box_a[3]
        bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[2], box_b[3]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    # ── Soft-NMS (Gaussian decay) ────────────────────────────────────

    def soft_nms(
        self,
        boxes: List[Tuple[float, float, float, float, float]],
        sigma: float = 0.5,
        score_cutoff: float = 0.10,
        max_boxes: int = 50,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Gaussian Soft-NMS: instead of hard-removing overlapping boxes,
        decay their scores proportionally to overlap.  This preserves
        nearby detections (e.g. two insects close together) while still
        suppressing true duplicates."""
        if not boxes:
            return []

        remaining = list(boxes)
        selected: List[Tuple[float, float, float, float, float]] = []

        while remaining and len(selected) < max_boxes:
            # Pick highest-scoring box
            remaining.sort(key=lambda b: b[4], reverse=True)
            best = remaining.pop(0)
            selected.append(best)

            # Decay scores of overlapping boxes
            new_remaining = []
            for box in remaining:
                iou = self.compute_iou(best, box)
                decayed_score = box[4] * math.exp(-(iou * iou) / sigma)
                if decayed_score >= score_cutoff:
                    new_remaining.append(
                        (box[0], box[1], box[2], box[3], decayed_score)
                    )
            remaining = new_remaining

        return selected

    # ── Hard NMS ─────────────────────────────────────────────────────

    def hard_nms(
        self,
        boxes: List[Tuple[float, float, float, float, float]],
        iou_threshold: float = 0.3,
        max_boxes: int = 5,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Strict greedy NMS: any box overlapping ≥iou_threshold with a
        higher-scoring box is removed entirely.  This enforces exactly
        one bounding box per insect."""
        if not boxes:
            return []

        sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        selected: List[Tuple[float, float, float, float, float]] = []

        for candidate in sorted_boxes:
            if len(selected) >= max_boxes:
                break
            keep = True
            for chosen in selected:
                if self.compute_iou(candidate, chosen) >= iou_threshold:
                    keep = False
                    break
            if keep:
                selected.append(candidate)

        return selected

    @staticmethod
    def is_oversized_box(x1, y1, x2, y2, image_width, image_height):
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        image_area = max(1.0, image_width * image_height)
        return (box_area / image_area) >= MAX_BOX_AREA_RATIO

    def handle_mouse_button_press_event(self, event_object):
        self.coordinate_x_where_mouse_press_started = event_object.x
        self.coordinate_y_where_mouse_press_started = event_object.y
        self.identifier_of_rectangle_currently_being_drawn = self.canvas_widget_for_drawing.create_rectangle(
            self.coordinate_x_where_mouse_press_started, self.coordinate_y_where_mouse_press_started, 
            self.coordinate_x_where_mouse_press_started, self.coordinate_y_where_mouse_press_started, 
            outline="red", width=2
        )
        self.boolean_flag_indicating_drawing_in_progress = True

    def handle_mouse_drag_event(self, event_object):
        if self.boolean_flag_indicating_drawing_in_progress and self.identifier_of_rectangle_currently_being_drawn:
            current_mouse_x, current_mouse_y = (event_object.x, event_object.y)
            self.canvas_widget_for_drawing.coords(
                self.identifier_of_rectangle_currently_being_drawn, 
                self.coordinate_x_where_mouse_press_started, 
                self.coordinate_y_where_mouse_press_started, 
                current_mouse_x, 
                current_mouse_y
            )

    def handle_mouse_button_release_event(self, event_object):
        if self.boolean_flag_indicating_drawing_in_progress and self.identifier_of_rectangle_currently_being_drawn:
            self.boolean_flag_indicating_drawing_in_progress = False
            coordinate_x_where_mouse_released, coordinate_y_where_mouse_released = (event_object.x, event_object.y)
            
            x_coordinate_1 = min(self.coordinate_x_where_mouse_press_started, coordinate_x_where_mouse_released)
            y_coordinate_1 = min(self.coordinate_y_where_mouse_press_started, coordinate_y_where_mouse_released)
            x_coordinate_2 = max(self.coordinate_x_where_mouse_press_started, coordinate_x_where_mouse_released)
            y_coordinate_2 = max(self.coordinate_y_where_mouse_press_started, coordinate_y_where_mouse_released)
            
            if (x_coordinate_2 - x_coordinate_1) > 5 and (y_coordinate_2 - y_coordinate_1) > 5:
                self.list_of_drawn_rectangles_data.append((x_coordinate_1, y_coordinate_1, x_coordinate_2, y_coordinate_2, self.identifier_of_rectangle_currently_being_drawn))
            else:
                self.canvas_widget_for_drawing.delete(self.identifier_of_rectangle_currently_being_drawn)
            
            self.identifier_of_rectangle_currently_being_drawn = None

    def handle_right_button_press_event(self, event_object):
        self.boolean_flag_deletion_selection_in_progress = True
        self.deletion_selection_start = (event_object.x, event_object.y)
        self.identifier_of_deletion_selection_rectangle = self.canvas_widget_for_drawing.create_rectangle(
            event_object.x,
            event_object.y,
            event_object.x,
            event_object.y,
            outline="blue",
            width=2,
            dash=(2, 2),
        )

    def handle_right_button_drag_event(self, event_object):
        if self.boolean_flag_deletion_selection_in_progress and self.identifier_of_deletion_selection_rectangle:
            start_x, start_y = self.deletion_selection_start
            self.canvas_widget_for_drawing.coords(
                self.identifier_of_deletion_selection_rectangle,
                start_x,
                start_y,
                event_object.x,
                event_object.y,
            )

    def handle_right_button_release_event(self, event_object):
        if not self.boolean_flag_deletion_selection_in_progress:
            return

        self.boolean_flag_deletion_selection_in_progress = False
        if not self.identifier_of_deletion_selection_rectangle:
            return

        x1, y1, x2, y2 = self.canvas_widget_for_drawing.coords(self.identifier_of_deletion_selection_rectangle)
        self.canvas_widget_for_drawing.delete(self.identifier_of_deletion_selection_rectangle)
        self.identifier_of_deletion_selection_rectangle = None

        sel_x1 = min(x1, x2)
        sel_y1 = min(y1, y2)
        sel_x2 = max(x1, x2)
        sel_y2 = max(y1, y2)

        remaining_rectangles = []
        for (rx1, ry1, rx2, ry2, rect_id) in self.list_of_drawn_rectangles_data:
            cx = (rx1 + rx2) / 2.0
            cy = (ry1 + ry2) / 2.0
            if sel_x1 <= cx <= sel_x2 and sel_y1 <= cy <= sel_y2:
                self.canvas_widget_for_drawing.delete(rect_id)
            else:
                remaining_rectangles.append((rx1, ry1, rx2, ry2, rect_id))

        self.list_of_drawn_rectangles_data = remaining_rectangles

    def save_annotations_to_file_and_proceed_to_next_image(self):
        identifier_for_current_class = self.dictionary_mapping_class_names_to_ids.get(self.name_of_currently_selected_folder, 0)
        original_image_width, original_image_height = self.original_image_object_from_pillow.size
        
        list_of_yolo_formatted_annotation_lines = []
        
        for (x_coordinate_1, y_coordinate_1, x_coordinate_2, y_coordinate_2, _) in self.list_of_drawn_rectangles_data:
            coordinate_x_1_on_original_image = x_coordinate_1 / self.scaling_ratio_applied_to_displayed_image
            coordinate_y_1_on_original_image = y_coordinate_1 / self.scaling_ratio_applied_to_displayed_image
            coordinate_x_2_on_original_image = x_coordinate_2 / self.scaling_ratio_applied_to_displayed_image
            coordinate_y_2_on_original_image = y_coordinate_2 / self.scaling_ratio_applied_to_displayed_image
            
            coordinate_x_1_on_original_image = max(0, min(original_image_width, coordinate_x_1_on_original_image))
            coordinate_y_1_on_original_image = max(0, min(original_image_height, coordinate_y_1_on_original_image))
            coordinate_x_2_on_original_image = max(0, min(original_image_width, coordinate_x_2_on_original_image))
            coordinate_y_2_on_original_image = max(0, min(original_image_height, coordinate_y_2_on_original_image))
            
            width_of_bounding_box = coordinate_x_2_on_original_image - coordinate_x_1_on_original_image
            height_of_bounding_box = coordinate_y_2_on_original_image - coordinate_y_1_on_original_image
            center_x_of_bounding_box = coordinate_x_1_on_original_image + (width_of_bounding_box / 2)
            center_y_of_bounding_box = coordinate_y_1_on_original_image + (height_of_bounding_box / 2)
            
            normalized_center_x = center_x_of_bounding_box / original_image_width
            normalized_center_y = center_y_of_bounding_box / original_image_height
            normalized_width = width_of_bounding_box / original_image_width
            normalized_height = height_of_bounding_box / original_image_height
            
            list_of_yolo_formatted_annotation_lines.append(f"{identifier_for_current_class} {normalized_center_x:.6f} {normalized_center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}")
            
        path_to_output_text_file = os.path.splitext(self.full_path_to_current_image_file)[0] + ".txt"
        with open(path_to_output_text_file, "w") as file_handle:
            file_handle.write("\n".join(list_of_yolo_formatted_annotation_lines))
            
        print(f"Saved {len(list_of_yolo_formatted_annotation_lines)} annotations for {os.path.basename(self.full_path_to_current_image_file)}")
        
        self.index_of_current_image_in_list += 1
        self.load_and_display_current_image_file()

    def delete_current_image_file_and_proceed_to_next_image(self):
        try:
            os.remove(self.full_path_to_current_image_file)
            print(f"Deleted {os.path.basename(self.full_path_to_current_image_file)}")
        except Exception as error_message:
            print(f"Error deleting file: {error_message}")
            
        self.index_of_current_image_in_list += 1
        self.load_and_display_current_image_file()

if __name__ == "__main__":
    root_tkinter_window = tk.Tk()
    application_instance = InsectImageAnnotationTool(root_tkinter_window)
    root_tkinter_window.mainloop()
