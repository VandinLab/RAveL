class dependencyInfos:
    # container class to have infos about independence tests

    def __init__(self, df, method="p-value", independence_method="cor", data_file="", indep_file="", save_every_update = True, independence_language = "R", calculus_type = "all"):
        self.method = method
        self.independence_method = independence_method
        self.data_file = data_file
        self.indep_file = indep_file
        self.df = df
        self.save_every_update = save_every_update
        self.save_counter = 0
        self.independence_language = independence_language
        self.calculus_type = calculus_type   # "all" "only_>=5" "all_>=5"

        self.lookout_PCD = {}
        self.lookout_PC = {}
        self.lookout_MB = {False: {}, True:{}}
        self.lookout_Mod_PCD = {}
        self.lookout_Mod_PC = {}
        self.lookout_MB_MOD = {False: {}, True:{}}
        self.lookout_PC_def = {}
        self.lookout_MB_algo = {}
        self.lookout_independence = {}
        self.lookout_RAveL_PC = {}
        self.lookout_RAveL_MB = {}
        self.lookout_independence = {}
        self.summands_initialized = False

