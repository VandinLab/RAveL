import os

test_type = 13

if not os.path.exists(os.path.join(os.getcwd(), "datasets")):
    os.mkdir(os.path.join(os.getcwd(), "datasets"))
path = os.path.join(os.getcwd(), "datasets","multiple"+str(test_type))
if not os.path.exists(path):
    os.mkdir(path)

if not os.path.exists(os.path.join(os.getcwd(), "IT_results")):
    os.mkdir(os.path.join(os.getcwd(), "IT_results"))
if not os.path.exists(os.path.join(os.getcwd(), "IT_results","total_res")):
    os.mkdir(os.path.join(os.getcwd(), "IT_results","total_res"))
if not os.path.exists(os.path.join(os.getcwd(), "IT_results","real_data")):
    os.mkdir(os.path.join(os.getcwd(), "IT_results","real_data"))
if not os.path.exists(os.path.join(os.getcwd(), "IT_results","multiple13")):
    os.mkdir(os.path.join(os.getcwd(), "IT_results","multiple13"))
if not os.path.exists(os.path.join(os.getcwd(), "pdfs")):
    os.mkdir(os.path.join(os.getcwd(), "pdfs"))