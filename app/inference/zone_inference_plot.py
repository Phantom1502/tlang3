from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from datasets import load_dataset

from app.config import AppConfig
from app.inference import ModelInference
from tlang import (
    TLangConfig,
    ProgramNode,
    Parser,
    ParseResult,
    plot_zones
)

class ZoneInferencePlot:
    def __init__(
        self,
        cfg: AppConfig,
        model_repo: str,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.cfg = cfg
        self.model = ModelInference(
            model_repo, 
            revision, 
            subfolder, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_p=top_p
        )
        self.tlang_cfg = TLangConfig(
            expected_candle_count=cfg.window.input_candles,
            n_bins=cfg.base.n_bins,
            digit_pad=cfg.base.digit_pad,
            mode="zone",
            zone_range=(cfg.base.zone_width_min_bins, cfg.base.zone_width_max_bins),
            sl_range=(50, 201),
            zone_extend_multiplier=1.0,
            last_n_touch=cfg.base.zone_last_n_touch
        )
    
    def _inference(
        self,
        prompt,
        n_samples: int = 8
    )-> List[str]:
        completions: List[str] = self.model.generate_samples(prompt, n_samples)
        zones = []
        input_candles = None
        for i in range(len(completions)):
            print(completions[i])
            parse_result: ParseResult = Parser.from_text(self.tlang_cfg, prompt + " " + completions[i]).parse()
            if parse_result.ast.think.zone is not None:
                zones.append(parse_result.ast.think.zone)
                input_candles = parse_result.ast.chart.candles
        
        plot_zones(input_candles, [], zones)
            
if __name__ == "__main__":
    from app.config import load_config
    cfg : AppConfig = load_config("./configs")
    model: ZoneInferencePlot = ZoneInferencePlot(cfg, "sullivan1502/base-zone-pretrain")
    model._inference("<chart> <O_936> <H_947> <L_930> <C_939> <O_938> <H_945> <L_930> <C_941> <O_947> <H_960> <L_938> <C_953> <O_952> <H_964> <L_948> <C_954> <O_955> <H_975> <L_954> <C_975> <O_975> <H_994> <L_967> <C_994> <O_994> <H_994> <L_975> <C_978> <O_978> <H_1036> <L_973> <C_1032> <O_1032> <H_1039> <L_1009> <C_1019> <O_1021> <H_1050> <L_1015> <C_1031> <O_1030> <H_1032> <L_1000> <C_1003> <O_1003> <H_1037> <L_1003> <C_1034> <O_1037> <H_1041> <L_1027> <C_1034> <O_1035> <H_1040> <L_1020> <C_1021> <O_1021> <H_1039> <L_1016> <C_1019> <O_1019> <H_1025> <L_1016> <C_1016> <O_1016> <H_1025> <L_1013> <C_1023> <O_1022> <H_1029> <L_1017> <C_1022> <O_1022> <H_1022> <L_995> <C_1016> <O_1015> <H_1021> <L_1011> <C_1015> <O_1014> <H_1034> <L_1009> <C_1033> <O_1034> <H_1044> <L_1033> <C_1039> <O_1040> <H_1042> <L_1029> <C_1036> <O_1036> <H_1056> <L_1035> <C_1040> <O_1040> <H_1041> <L_1030> <C_1036> <O_1037> <H_1050> <L_1035> <C_1045> <O_1045> <H_1046> <L_999> <C_1002> <O_1003> <H_1003> <L_979> <C_979> <O_979> <H_991> <L_977> <C_988> <O_989> <H_1012> <L_974> <C_1012> <O_1012> <H_1016> <L_1001> <C_1006> <O_1006> <H_1014> <L_998> <C_1013> <O_1012> <H_1016> <L_996> <C_1013> <O_1013> <H_1018> <L_978> <C_980> <O_980> <H_985> <L_964> <C_984> <O_984> <H_1000> <L_981> <C_996> <O_996> <H_1006> <L_988> <C_996> <O_996> <H_1004> <L_967> <C_973> <O_974> <H_988> <L_974> <C_986> <O_987> <H_987> <L_967> <C_972> <O_973> <H_976> <L_965> <C_974> <O_974> <H_974> <L_960> <C_970> <O_968> <H_970> <L_948> <C_955> <O_955> <H_964> <L_946> <C_961> <O_962> <H_995> <L_961> <C_988> <O_989> <H_1003> <L_981> <C_982> <O_983> <H_993> <L_983> <C_984> <O_984> <H_995> <L_980> <C_994> <O_994> <H_1000> <L_969> <C_974> <O_974> <H_979> <L_957> <C_957> <O_957> <H_988> <L_943> <C_980> <O_981> <H_981> <L_967> <C_971> <O_972> <H_973> <L_947> <C_948> <O_948> <H_952> <L_932> <C_936> <O_936> <H_945> <L_912> <C_913> <O_913> <H_925> <L_900> <C_923> <O_923> <H_937> <L_915> <C_937> <O_936> <H_959> <L_930> <C_958> <O_958> <H_973> <L_948> <C_967> <O_967> <H_967> <L_944> <C_950> <O_950> <H_963> <L_948> <C_956> <O_956> <H_956> <L_940> <C_951> <O_951> <H_959> <L_937> <C_955> <O_956> <H_970> <L_952> <C_967> <O_968> <H_968> <L_938> <C_950> <O_949> <H_959> <L_948> <C_956> <O_957> <H_962> <L_950> <C_961> <O_959> <H_1001> <L_956> <C_999> <O_998> <H_1064> <L_997> <C_1050> <O_1052> <H_1065> <L_1037> <C_1039> <O_1040> <H_1063> <L_1034> <C_1055> <O_1058> <H_1068> <L_1047> <C_1060> <O_1060> <H_1065> <L_1042> <C_1052> <O_1052> <H_1084> <L_1051> <C_1053> <O_1053> <H_1056> <L_1022> <C_1028> <O_1028> <H_1040> <L_1023> <C_1033> <O_1033> <H_1046> <L_1023> <C_1032> <O_1032> <H_1033> <L_1016> <C_1018> <O_1018> <H_1046> <L_1018> <C_1042> <O_1042> <H_1046> <L_1028> <C_1036> <O_1037> <H_1040> <L_1027> <C_1033> <O_1033> <H_1044> <L_1030> <C_1042> <O_1042> <H_1046> <L_1023> <C_1032> <O_1031> <H_1033> <L_1020> <C_1031> <O_1031> <H_1059> <L_1026> <C_1055> <O_1055> <H_1069> <L_1054> <C_1067> <O_1067> <H_1071> <L_1057> <C_1065> <O_1065> <H_1085> <L_1063> <C_1082> <O_1082> <H_1104> <L_1077> <C_1096> <O_1096> <H_1108> <L_1074> <C_1099> <O_1099> <H_1107> <L_1068> <C_1068> <O_1068> <H_1075> <L_1060> <C_1060> <O_1059> <H_1067> <L_1021> <C_1025> <O_1024> <H_1044> <L_1016> <C_1043> <O_1042> <H_1046> <L_1027> <C_1035> <O_1035> <H_1052> <L_1024> <C_1047> <O_1047> <H_1049> <L_1029> <C_1029> <O_1031> <H_1113> <L_1028> <C_1113> <O_1113> <H_1132> <L_1102> <C_1128> <O_1128> <H_1147> <L_1120> <C_1122> </chart>")