"""
主 Pipeline 模块
整合 ASR、语法修复、呼号修复，输出结构化 JSON
"""
import json
import re
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from asr_infer import ASRInference
from atc_grammar import ATCGrammarCorrector
from callsign_fix import CallsignFixer
from utils.config import Config


class ATCASRPipeline:
    """ATC ASR 完整 Pipeline"""

    def __init__(
        self,
        device: str = "cuda:0",
        use_hotwords: bool = True,
    ):
        """
        初始化 Pipeline

        Args:
            device: 设备 (cuda:0 或 cpu)
            use_hotwords: 是否使用热词
        """
        self.config = Config()
        self.use_hotwords = use_hotwords

        logger.info("初始化 ATC ASR Pipeline")

        # 初始化各模块
        self.asr = ASRInference(device=device)
        self.grammar_corrector = ATCGrammarCorrector()
        self.callsign_fixer = CallsignFixer()

        logger.info("Pipeline 初始化完成")

    def parse_command_type(self, text: str) -> List[str]:
        """
        识别指令类型
        """
        command_types = []

        for cmd_type, keywords in self.config.COMMAND_TYPES.items():
            for keyword in keywords:
                if keyword in text:
                    command_types.append(cmd_type)
                    break

        return command_types

    def extract_parameters(self, text: str) -> Dict[str, Any]:
        """
        提取指令参数（高度、航向、频率等）
        """
        params = {}

        # 提取高度
        altitude_match = re.search(r'高度(\d+)', text)
        if altitude_match:
            params["altitude"] = int(altitude_match.group(1))

        # 提取航向
        heading_match = re.search(r'航向(\d+)', text)
        if heading_match:
            params["heading"] = int(heading_match.group(1))

        # 提取频率
        frequency_match = re.search(r'频率(\d+\.?\d*)', text)
        if frequency_match:
            params["frequency"] = float(frequency_match.group(1))

        # 提取跑道
        runway_match = re.search(r'跑道(\d+[LRC]?)', text)
        if runway_match:
            params["runway"] = runway_match.group(1)

        # 提取速度
        speed_match = re.search(r'速度(\d+)', text)
        if speed_match:
            params["speed"] = int(speed_match.group(1))

        return params

    def parse_commands(self, text: str) -> List[Dict[str, Any]]:
        """
        解析具体指令
        """
        commands = []
        params = self.extract_parameters(text)

        # 航向指令
        if "heading" in params:
            if "右转" in text:
                commands.append({
                    "type": "heading_change",
                    "action": "右转",
                    "heading": params["heading"]
                })
            elif "左转" in text:
                commands.append({
                    "type": "heading_change",
                    "action": "左转",
                    "heading": params["heading"]
                })
            else:
                commands.append({
                    "type": "heading_change",
                    "action": "航向",
                    "heading": params["heading"]
                })

        # 高度指令
        if "altitude" in params:
            if "爬升" in text or "上升" in text:
                commands.append({
                    "type": "altitude_change",
                    "action": "爬升",
                    "altitude": params["altitude"]
                })
            elif "下降" in text:
                commands.append({
                    "type": "altitude_change",
                    "action": "下降",
                    "altitude": params["altitude"]
                })
            elif "保持" in text:
                commands.append({
                    "type": "altitude_change",
                    "action": "保持",
                    "altitude": params["altitude"]
                })

        # 频率指令
        if "frequency" in params:
            commands.append({
                "type": "frequency_change",
                "action": "联系" if "联系" in text else "切换",
                "frequency": params["frequency"]
            })

        # 起降指令
        if "起飞" in text:
            commands.append({
                "type": "takeoff",
                "action": "起飞",
                "runway": params.get("runway")
            })
        elif "着陆" in text or "落地" in text:
            commands.append({
                "type": "landing",
                "action": "着陆",
                "runway": params.get("runway")
            })
        elif "复飞" in text:
            commands.append({
                "type": "go_around",
                "action": "复飞"
            })

        # 滑行指令
        if "滑行" in text or "滑出" in text:
            commands.append({
                "type": "taxi",
                "action": "滑行"
            })

        return commands

    def _load_ground_truth(self, audio_path: Union[str, Path]) -> Optional[str]:
        """
        加载对应的 ground truth 文本

        Args:
            audio_path: 音频文件路径

        Returns:
            ground truth 文本，如果不存在则返回 None
        """
        audio_path = Path(audio_path)

        # 构建对应的文本文件路径
        # WAVdata/ch01/ch01_01/ch01_01_001.wav -> TXTdata/ch01/ch01_01/ch01_01_001.txt
        txt_path = str(audio_path).replace("WAVdata", "TXTdata").replace(".wav", ".txt")
        txt_path = Path(txt_path)

        if txt_path.exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # 移除可能的行号前缀 (如 "1→文本")
                    if '→' in content:
                        content = content.split('→', 1)[1]
                    return content
            except Exception as e:
                logger.warning(f"读取 ground truth 失败: {e}")
                return None
        else:
            logger.debug(f"Ground truth 文件不存在: {txt_path}")
            return None

    def process(
        self,
        audio_input: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        处理单个音频文件

        Args:
            audio_input: 音频文件路径

        Returns:
            结构化 JSON 输出
        """
        logger.info(f"处理音频: {audio_input}")

        # 0. 加载 ground truth
        ground_truth = self._load_ground_truth(audio_input)

        # 1. ASR 识别
        raw_text = self.asr.transcribe(audio_input, use_hotwords=self.use_hotwords)
        logger.info(f"ASR 识别: {raw_text}")

        # 2. 语法修复
        normalized_text = self.grammar_corrector.normalize(raw_text)
        logger.info(f"语法修复: {normalized_text}")

        # 3. 呼号修复
        fixed_text = self.callsign_fixer.fix_callsign(normalized_text)
        logger.info(f"呼号修复: {fixed_text}")

        # 4. 提取呼号
        callsigns = self.callsign_fixer.extract_all_callsigns(fixed_text)
        callsign = callsigns[0] if callsigns else None

        # 5. 识别指令类型
        command_types = self.parse_command_type(fixed_text)

        # 6. 解析具体指令
        commands = self.parse_commands(fixed_text)

        # 7. 构建输出
        output = {
            "audio_file": str(audio_input),
            "timestamp": datetime.now().isoformat(),
            "ground_truth": ground_truth,
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "final_text": fixed_text,
            "callsign": callsign,
            "command_types": command_types,
            "commands": commands,
        }

        return output

    def process_batch(
        self,
        audio_inputs: List[Union[str, Path]],
    ) -> List[Dict[str, Any]]:
        """
        批量处理音频文件

        Args:
            audio_inputs: 音频文件路径列表

        Returns:
            结构化 JSON 输出列表
        """
        logger.info(f"批量处理 {len(audio_inputs)} 个音频文件")

        results = []
        for audio_input in audio_inputs:
            try:
                result = self.process(audio_input)
                results.append(result)
            except Exception as e:
                logger.error(f"处理失败 {audio_input}: {e}")
                results.append({
                    "audio_file": str(audio_input),
                    "error": str(e)
                })

        return results


def main():
    """测试函数"""
    # 初始化 Pipeline
    pipeline = ATCASRPipeline(device="cuda:0")

    # 测试音频
    config = Config()
    audio_dir = config.AUDIO_DIR

    test_audio = list(audio_dir.glob("**/*.wav"))[:3]

    if test_audio:
        logger.info(f"找到 {len(test_audio)} 个测试音频")

        # 单个处理
        for audio_path in test_audio:
            result = pipeline.process(audio_path)
            print("\n" + "=" * 80)
            print(json.dumps(result, ensure_ascii=False, indent=2))

        # 批量处理
        results = pipeline.process_batch(test_audio)
        print("\n" + "=" * 80)
        print("批量处理结果:")
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        logger.warning("未找到测试音频文件")


if __name__ == "__main__":
    main()
