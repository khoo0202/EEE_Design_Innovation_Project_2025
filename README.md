##############################################################
#                                                            #
#                🧠  S T A R T E R   G U I D E  🚀    	     #
#                                                            #
#   Smoking/Vaping Detection System using YOLO + VQA + UI    #
#                                                            #
#   1️ Create & activate a virtual environment        	     #
#       py -3.12 -m venv .venv                                 #
#       .venv\Scripts\activate   (Windows)                 #
#       source .venv/bin/activate    (macOS/Linux)           #
#                                                            #
#   2️ Install dependencies                           	     #
#       pip install -r requirements.txt                      #
#       if ERROR: No matching distribution found for torch==2.5.1+cu121
#              if ERROR: No matching distribution found for torch==2.5.1+cu121
#       pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1                         #
#   3️ Launch Ollama backend and download model        	     #
#       ollama serve                                         #
#       ollama pull qwen2.5vl:3b                             #
#                                                            #
#   4️ Run the VQA server                             	     #
#       python vqa_newest_desc_strictness.py                 #
#                                                            #
#   5️ Start the app (UI / main pipeline)              	     #
#       python main_app.py                                   #
#                                                            #
#   6️ Train and test calibration model (optional)   	     #
#       python -m tools.train_logreg --root ./dataset        #
#       python -m tools.test_logreg  --root ./dataset        #
#                                                            #
#   Dataset structure:                                       #
#     dataset/                                               #
#       train/{smoking,vaping,none}                          #
#       val/{smoking,vaping,none}                            #
#       test/{smoking,vaping,none}                           #
#                                                            #
#   ✅  After setup: open your UI or check ./out/ for logs.  #
#                                                            #
##############################################################