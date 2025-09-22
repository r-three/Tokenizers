#!/usr/bin/env python3
"""
Comprehensive Tokenizer Analysis Script
Analyzes multiple tokenizers across different languages and properties
"""

import json
import os
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

from xarch_tokenizers.scripts.tokenizer_card.supertoken import (
    MistralTokenizer,
    TokenMonsterTokenizer,
)

warnings.filterwarnings('ignore')

# Required imports - install with: pip install transformers tiktoken tokenizers sentencepiece
try:
    import tiktoken
    from transformers import AutoModel, AutoTokenizer

    import tokenizers
except ImportError as e:
    print(f"Missing required libraries. Install with: pip install transformers tiktoken tokenizers sentencepiece")
    raise e

class TokenizerAnalyzer:
    def __init__(self):
        self.test_languages = {
            'english': {
                'text': 'Hello world! This is a test sentence with numbers 12345 and symbols @#$%.',
                'unicode_test': 'café naïve résumé',
                'oov_test': 'supercalifragilisticexpialidocious'
            },
            'farsi': {
                'text': 'سلام دنیا! این یک جمله آزمایشی با اعداد ۱۲۳۴۵ است.',
                'unicode_test': 'پیش‌بینی می‌کنم که این متن درست نمایش داده شود.',
                'oov_test': 'نیوموناولترامیکروسکوپیک‌سیلیکوولکانوکونیوزیس'
            },
            'turkish': {
                'text': 'Merhaba dünya! Bu sayılar 12345 içeren bir test cümlesidir.',
                'unicode_test': 'Türkçe karakterler: ğüşıöç ĞÜŞIÖÇ',
                'oov_test': 'muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine'
            },
            'chinese': {
                'text': '你好世界！这是一个包含数字12345的测试句子。',
                'unicode_test': '繁体字测试：語言處理技術',
                'oov_test': '超长复合词测试语言模型分析'
            },
            'italian': {
                'text': 'Ciao mondo! Questa è una frase di test con numeri 12345.',
                'unicode_test': 'Caratteri accentati: àèìòù éáíóú',
                'oov_test': 'precipitevolissimevolmente'
            }
        }
        
        self.tokenizer_configs = {
            # 'aya-expanse': {'model_name': 'CohereForAI/aya-expanse-8b'},
            # 'bigscience-bloom': {'model_name': 'bigscience/bloom-560m'},
            # 'common-pile-comma-v0.1': {'model_name': 'EleutherAI/gpt-neox-20b'},  # Uses same tokenizer
            # 'google-bert-bert-base-multilingual-cased': {'model_name': 'google-bert/bert-base-multilingual-cased'},
            # 'google-byt5-small': {'model_name': 'google/byt5-small'},
            # 'google-gemma-2-2b': {'model_name': 'google/gemma-2-2b'},
            # 'gpt2': {'model_name': 'gpt2'},
            # 'meta-llama-Llama-3.2-1B': {'model_name': 'meta-llama/Llama-3.2-1B'},
            # 'microsoft-Phi-3-mini-4k-instruct': {'model_name': 'microsoft/Phi-3-mini-4k-instruct'},
            'mistralai-tekken': {'is_tekken': True},  # Tekken tokenizer
            # 'Qwen-Qwen3-8B': {'model_name': 'Qwen/Qwen2.5-7B'},
            # 'tiktoken-gpt-4o': {'is_tiktoken': True, 'encoding': 'o200k_base'},
            'tokenmonster-englishcode-32000-consistent-v1': {'is_tokenmonster': True, 'model_name': 'englishcode-32000-consistent-v1'},
            # 'facebook-xglm-564m': {'model_name': 'facebook/xglm-564M'}
        }
        
        self.results = []

    def load_tokenizer(self, tokenizer_name: str) -> Optional[Any]:
        """Load tokenizer based on configuration"""
        config = self.tokenizer_configs.get(tokenizer_name, {})
        
        if config.get('skip'):
            print(f"Skipping {tokenizer_name}: {config.get('reason', 'Not supported')}")
            return None
            
        try:
            if config.get('is_tiktoken'):
                return tiktoken.get_encoding(config['encoding'])
            elif config.get('is_tekken'):
                return MistralTokenizer.load("")
            elif config.get('is_tokenmonster'):
                return TokenMonsterTokenizer.load(config.get('model_name'))
            else:
                model_name = config.get('model_name')
                if not model_name:
                    print(f"No model name configured for {tokenizer_name}")
                    return None
                return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load {tokenizer_name}: {str(e)}")
            return None

    def analyze_tokenizer(self, tokenizer_name: str, tokenizer) -> Dict[str, Any]:
        """Comprehensive analysis of a single tokenizer"""
        print(f"\nAnalyzing {tokenizer_name}...")
        
        result = {
            'tokenizer_name': tokenizer_name,
            'inherits_from': self.get_inheritance(tokenizer_name, tokenizer),
            'diff_from_parent': '',  # To be filled manually based on inheritance
            'tokenization_method': self.get_tokenization_method(tokenizer_name, tokenizer),
            'vocab_size': self.get_vocab_size(tokenizer),
            'special_characters': self.get_special_tokens(tokenizer),
            'byte_fallback': self.check_byte_fallback(tokenizer),
            'multi_lingual': self.test_multilingual_support(tokenizer),
            'numbers': self.test_number_handling(tokenizer),
            'pretokenization_splits': self.analyze_pretokenization(tokenizer),
            'unicode_normalization': self.test_unicode_normalization(tokenizer),
            'continuation': self.test_continuation_tokens(tokenizer),
            'oov_handling': self.test_oov_handling(tokenizer),
            'script_coverage': self.test_script_coverage(tokenizer),
            'whitespace_treatment': self.test_whitespace_treatment(tokenizer),
            'release_date': self.get_release_date(tokenizer_name),
            # Language-specific details
            'language_analysis': self.analyze_languages(tokenizer),
            # Comprehensive pretokenization analysis
            'pretokenization_detailed': self.comprehensive_pretokenization_analysis(tokenizer)
        }
        
        return result

    def get_inheritance(self, tokenizer_name: str, tokenizer) -> str:
        """Determine tokenizer inheritance"""
        #TODO
        inheritance_map = {
            # 'aya-expanse': 'Llama/SentencePiece',
            # 'bigscience-bloom': 'GPT-2/BPE',
            # 'common-pile-comma-v0.1': 'GPT-NeoX/BPE',
            # 'google-bert-bert-base-multilingual-cased': 'BERT/WordPiece',
            # 'google-byt5-small': 'T5/SentencePiece (Byte-level)',
            # 'google-gemma-2-2b': 'SentencePiece',
            # 'gpt2': 'Base BPE',
            # 'meta-llama-Llama-3.2-1B': 'Llama/SentencePiece',
            # 'microsoft-Phi-3-mini-4k-instruct': 'Llama/SentencePiece',
            # 'mistralai-tekken': 'Tekken/Custom',
            # 'Qwen-Qwen3-8B': 'tiktoken/BPE',
            # 'tiktoken-gpt-4o': 'tiktoken/BPE',
            # 'facebook-xglm-564m': 'SentencePiece'
        }
        return inheritance_map.get(tokenizer_name, 'Unknown')

    def get_tokenization_method(self, tokenizer_name: str, tokenizer) -> str:
        """Identify tokenization method"""
        if hasattr(tokenizer, 'backend_tokenizer'):
            backend = str(type(tokenizer.backend_tokenizer))
            if 'BPE' in backend:
                return 'BPE (Byte-Pair Encoding)'
            elif 'WordPiece' in backend:
                return 'WordPiece'
            elif 'SentencePiece' in backend:
                return 'SentencePiece'
        
        method_map = {
            'tiktoken': 'BPE (tiktoken variant)',
            'byt5': 'Byte-level SentencePiece',
            'bert': 'WordPiece',
            'gpt': 'BPE'
        }
        
        for key, method in method_map.items():
            if key in tokenizer_name.lower():
                return method
        
        return 'Unknown'

    def get_vocab_size(self, tokenizer) -> int:
        """Get vocabulary size"""
        try:
            if hasattr(tokenizer, 'vocab_size'):
                return tokenizer.vocab_size
            elif hasattr(tokenizer, 'n_vocab'):
                return tokenizer.n_vocab
            elif hasattr(tokenizer, '__len__'):
                return len(tokenizer)
            else:
                return -1
        except:
            return -1

    def get_special_tokens(self, tokenizer) -> str:
        """Extract special tokens"""
        special_tokens = []
        
        try:
            if hasattr(tokenizer, 'special_tokens_map'):
                special_tokens.extend(tokenizer.special_tokens_map.values())
            
            # Common special tokens to check
            attrs = ['pad_token', 'eos_token', 'bos_token', 'unk_token', 'sep_token', 'cls_token', 'mask_token']
            for attr in attrs:
                if hasattr(tokenizer, attr):
                    token = getattr(tokenizer, attr)
                    if token and token not in special_tokens:
                        special_tokens.append(token)
                        
        except Exception as e:
            special_tokens.append(f"Error extracting: {str(e)}")
            
        return ', '.join([str(token) for token in special_tokens[:10]])  # Limit output

    def check_byte_fallback(self, tokenizer) -> bool:
        """Check if tokenizer has byte-level fallback"""
        try:
            # Test with a very rare character
            test_char = '𝕳'  # Mathematical bold H
            tokens = tokenizer.encode(test_char)
            decoded = tokenizer.decode(tokens)
            return test_char == decoded
        except:
            return False

    def test_multilingual_support(self, tokenizer) -> str:
        """Test multilingual support quality"""
        scores = {}
        
        for lang, tests in self.test_languages.items():
            try:
                text = tests['text']
                tokens = tokenizer.encode(text)
                decoded = tokenizer.decode(tokens)
                
                # Simple quality metrics
                char_preservation = len(decoded) / len(text) if len(text) > 0 else 0
                token_efficiency = len(text) / len(tokens) if len(tokens) > 0 else 0
                
                scores[lang] = {
                    'tokens': len(tokens),
                    'preservation': char_preservation,
                    'efficiency': token_efficiency
                }
            except Exception as e:
                scores[lang] = {'error': str(e)}
        
        # Determine overall multilingual capability
        working_langs = [lang for lang, score in scores.items() if 'error' not in score]
        return f"Supports {len(working_langs)}/5 languages tested"

    def test_number_handling(self, tokenizer) -> str:
        """Test how numbers are tokenized"""
        test_numbers = ['123', '12345', '3.14159', '1,000,000']
        results = []
        
        for num in test_numbers:
            try:
                tokens = tokenizer.encode(num)
                results.append(f"{num}→{len(tokens)}tok")
            except:
                results.append(f"{num}→error")
                
        return '; '.join(results)

    def analyze_pretokenization(self, tokenizer) -> str:
        """Analyze pretokenization behavior - basic version for summary"""
        try:
            analysis = self.comprehensive_pretokenization_analysis(tokenizer)
            # Return a brief summary for the main table
            strategy = analysis.get('primary_strategy', 'unknown')
            splits = analysis.get('splits_on', [])
            return f"{strategy}; splits: {', '.join(splits[:3])}"
        except:
            return "unknown"

    def test_unicode_normalization(self, tokenizer) -> str:
        """Test Unicode normalization behavior"""
        # Test with composed vs decomposed Unicode
        composed = "é"  # single character
        decomposed = "é"  # e + combining acute
        
        try:
            tokens_c = tokenizer.encode(composed)
            tokens_d = tokenizer.encode(decomposed)
            
            if tokens_c == tokens_d:
                return "NFC normalization"
            else:
                return "preserves Unicode form"
        except:
            return "unknown"

    def test_continuation_tokens(self, tokenizer) -> str:
        """Test for continuation/prefix tokens"""
        try:
            tokens = tokenizer.encode("hello world")
            decoded_tokens = [tokenizer.decode([token]) for token in tokens]
            
            has_continuation = any(token.startswith('##') or token.startswith('▁') or token.startswith('Ġ') 
                                 for token in decoded_tokens)
            
            if has_continuation:
                return "uses continuation markers"
            else:
                return "no continuation markers"
        except:
            return "unknown"

    def test_oov_handling(self, tokenizer) -> str:
        """Test Out-of-Vocabulary handling"""
        strategies = []
        
        for lang, tests in self.test_languages.items():
            try:
                oov_text = tests['oov_test']
                tokens = tokenizer.encode(oov_text)
                decoded = tokenizer.decode(tokens)
                
                if decoded == oov_text:
                    strategies.append("subword fallback")
                elif len(decoded) < len(oov_text):
                    strategies.append("truncation")
                else:
                    strategies.append("approximation")
                    
            except Exception as e:
                strategies.append("error")
                
        return "; ".join(set(strategies))

    def test_script_coverage(self, tokenizer) -> str:
        """Test coverage of different scripts"""
        scripts = {
            'Latin': 'Hello',
            'Arabic': 'سلام',
            'Chinese': '你好',
            'Cyrillic': 'Привет'
        }
        
        supported = []
        for script, text in scripts.items():
            try:
                tokens = tokenizer.encode(text)
                decoded = tokenizer.decode(tokens)
                if decoded == text:
                    supported.append(script)
            except:
                pass
                
        return ', '.join(supported)

    def test_whitespace_treatment(self, tokenizer) -> str:
        """Test how whitespace is handled"""
        tests = {
            'single_space': 'hello world',
            'multiple_spaces': 'hello    world',
            'tabs': 'hello\tworld',
            'newlines': 'hello\nworld'
        }
        
        behaviors = []
        for test_name, text in tests.items():
            try:
                tokens = tokenizer.encode(text)
                decoded = tokenizer.decode(tokens)
                
                if decoded == text:
                    behaviors.append(f"{test_name}:preserved")
                else:
                    behaviors.append(f"{test_name}:modified")
            except:
                behaviors.append(f"{test_name}:error")
                
        return '; '.join(behaviors)

    def get_release_date(self, tokenizer_name: str) -> str:
        """Approximate release dates based on model knowledge"""
        # not confirmed
        dates = {
            # 'gpt2': '2019-02',
            # 'google-bert-bert-base-multilingual-cased': '2018-10',
            # 'google-byt5-small': '2021-05',
            # 'bigscience-bloom': '2022-07',
            # 'meta-llama-Llama-3.2-1B': '2024-09',
            # 'microsoft-Phi-3-mini-4k-instruct': '2024-04',
            # 'tiktoken-gpt-4o': '2024-05',
            # 'google-gemma-2-2b': '2024-06',
            # 'Qwen-Qwen3-8B': '2024-09',
            # 'facebook-xglm-564m': '2021-12',
            # 'aya-expanse': '2024-11',
            # 'mistralai-tekken': '2024-10'
        }
        return dates.get(tokenizer_name, 'Unknown')

    def analyze_languages(self, tokenizer) -> Dict[str, Dict]:
        """Detailed analysis for each target language"""
        analysis = {}
        
        for lang, tests in self.test_languages.items():
            lang_result = {}
            
            for test_type, text in tests.items():
                try:
                    tokens = tokenizer.encode(text)
                    decoded = tokenizer.decode(tokens)
                    
                    lang_result[test_type] = {
                        'original_length': len(text),
                        'token_count': len(tokens),
                        'compression_ratio': len(text) / len(tokens) if len(tokens) > 0 else 0,
                        'perfect_reconstruction': text == decoded,
                        'tokens_per_char': len(tokens) / len(text) if len(text) > 0 else 0
                    }
                except Exception as e:
                    lang_result[test_type] = {'error': str(e)}
                    
            analysis[lang] = lang_result
            
        return analysis

    def comprehensive_pretokenization_analysis(self, tokenizer) -> Dict[str, Any]:
        """Deep dive into pretokenization patterns and strategies"""
        analysis = {
            'primary_strategy': 'unknown',
            'splits_on': [],
            'preserves': [],
            'normalization_effects': {},
            'boundary_handling': {},
            'special_cases': {},
            'pattern_examples': {},
            'cross_linguistic_behavior': {}
        }
        
        # Test cases for different pretokenization aspects
        test_cases = {
            # Basic splitting patterns
            'whitespace_basic': 'hello world test',
            'whitespace_multiple': 'hello    world',
            'whitespace_mixed': 'hello\tworld\ntest',
            
            # Punctuation handling
            'punctuation_attached': 'hello,world!test.',
            'punctuation_spaced': 'hello , world ! test .',
            'punctuation_mixed': 'hello, world!',
            'quotes_and_brackets': '"hello" (world) [test]',
            'apostrophes': "don't can't won't it's",
            'hyphens': 'state-of-the-art self-aware',
            
            # Numbers and special characters
            'numbers_standalone': '123 456 789',
            'numbers_mixed': 'test123 456test test456test',
            'decimal_numbers': '3.14159 2.5 100.0',
            'currency': '$100 €50 ¥1000',
            'percentages': '50% 3.14% 100%',
            'dates_times': '2024-01-01 12:30:45',
            'urls_emails': 'https://example.com user@email.com',
            
            # Contractions and apostrophes
            'contractions_basic': "I'm you're they've we'll",
            'contractions_unicode': "I'm you're they've we'll",  # Unicode apostrophe U+2019
            'possessives': "John's car dogs' toys",
            # 'french_contractions': "c'est n'est l'eau",
            'italian_elision': "dell'arte all'italiana",
            
            # Case sensitivity  
            'mixed_case': 'Hello WORLD Test',
            'camelCase': 'camelCase PascalCase',
            'acronyms': 'USA NASA FBI',
            
            # Diacritics and accents
            'latin_diacritics': 'café naïve résumé piñata',
            'combining_diacritics': 'cafe\u0301 nai\u0308ve',  # Decomposed forms
            'turkish_special': 'İstanbul Güneş ğüşıöç',
            'vietnamese_tones': 'tiếng việt',
            'scandinavian': 'høj kött Åse',
            
            # Unicode normalization forms
            'nfc_form': 'café naïve',  # NFC (composed)
            'nfd_form': 'cafe\u0301 nai\u0308ve',  # NFD (decomposed)
            
            # Special characters and symbols
            'symbols': '@#$%^&*()+=',
            'unicode_symbols': '→←↑↓★☆♠♥',
            'mathematical': '∑∏∆∇∞±',
            'emojis': '😀😃😄😁',
            
            # Programming/technical content
            'code_like': 'function_name variable_x class.method()',
            'file_paths': '/path/to/file.txt C:\\Windows\\System32',
            'xml_html': '<tag>content</tag> <div class="test">',
            
            # Language-specific challenges
            'contractions': "I'm you're they've we'll",
            'possessives': "John's car the dogs' toys",
            'compound_words': 'twenty-first state-of-the-art',
            
            # Whitespace edge cases
            'leading_trailing': '  hello world  ',
            'zero_width': 'hello\u200bworld',  # Zero-width space
            'non_breaking': 'hello\u00a0world',  # Non-breaking space
        }
        
        # Add multilingual test cases
        multilingual_tests = {
            'chinese_no_spaces': '你好世界这是测试',
            'japanese_mixed': 'こんにちは世界Hello',
            'arabic_rtl': 'مرحبا بالعالم اختبار',
            'thai_no_spaces': 'สวัสดีโลกทดสอบ',
            'korean_mixed': '안녕하세요 world 테스트',
            'hindi_devanagari': 'नमस्ते दुनिया परीक्षण',
            'persian_farsi': 'سلام دنیا آزمایش',
            'turkish_special': 'merhaba dünya İÜÖÇŞ ğüşıöç',
        }
        test_cases.update(multilingual_tests)
        
        try:
            # Analyze each test case
            for test_name, test_text in test_cases.items():
                try:
                    tokens = tokenizer.encode(test_text)
                    if len(tokens) == 0:
                        continue
                        
                    # Get individual token representations
                    decoded_tokens = []
                    for token in tokens:
                        try:
                            decoded = tokenizer.decode([token])
                            decoded_tokens.append(decoded)
                        except:
                            decoded_tokens.append(f"<token_{token}>")
                    
                    analysis['pattern_examples'][test_name] = {
                        'input': test_text,
                        'tokens': decoded_tokens,
                        'token_count': len(tokens),
                        'char_to_token_ratio': len(test_text) / len(tokens) if len(tokens) > 0 else 0
                    }
                    
                except Exception as e:
                    analysis['pattern_examples'][test_name] = {'error': str(e)}
            
            # Analyze splitting patterns
            analysis.update(self._analyze_splitting_patterns(analysis['pattern_examples']))
            
            # Determine primary strategy
            analysis['primary_strategy'] = self._determine_primary_strategy(analysis)
            
            # Analyze boundary handling
            analysis['boundary_handling'] = self._analyze_boundary_handling(analysis['pattern_examples'])
            
            # Cross-linguistic behavior
            analysis['cross_linguistic_behavior'] = self._analyze_cross_linguistic_behavior(analysis['pattern_examples'])
            
            # Analyze normalization effects
            analysis['normalization_effects'] = self._analyze_normalization_effects(tokenizer)
            
        except Exception as e:
            analysis['error'] = str(e)
            
        return analysis
    
    def _analyze_splitting_patterns(self, examples: Dict) -> Dict[str, Any]:
        """Analyze what the tokenizer splits on"""
        splits_on = []
        preserves = []
        
        # Check whitespace handling
        if 'whitespace_basic' in examples and 'tokens' in examples['whitespace_basic']:
            tokens = examples['whitespace_basic']['tokens']
            if len(tokens) > 1:
                splits_on.append('spaces')
                # Check if spaces are preserved
                if any(' ' in token for token in tokens):
                    preserves.append('spaces_in_tokens')
                else:
                    preserves.append('space_splitting')
        
        # Check punctuation handling
        if 'punctuation_attached' in examples and 'tokens' in examples['punctuation_attached']:
            tokens = examples['punctuation_attached']['tokens']
            punct_tokens = [t for t in tokens if any(p in t for p in ',.!?;:')]
            if len(punct_tokens) > 0:
                if any(len(t) == 1 and t in ',.!?;:' for t in tokens):
                    splits_on.append('punctuation')
                else:
                    preserves.append('punctuation_attached')
        
        # Check number handling
        if 'numbers_mixed' in examples and 'tokens' in examples['numbers_mixed']:
            tokens = examples['numbers_mixed']['tokens']
            if any(any(c.isdigit() for c in token) for token in tokens):
                preserves.append('numbers')
        
        # Check case sensitivity
        if 'mixed_case' in examples and 'tokens' in examples['mixed_case']:
            tokens = examples['mixed_case']['tokens']
            original = examples['mixed_case']['input']
            if any(token.lower() != token.upper() for token in tokens if token.isalpha()):
                preserves.append('case')
        
        # Check special character handling
        if 'symbols' in examples and 'tokens' in examples['symbols']:
            tokens = examples['symbols']['tokens']
            if len(tokens) > 1:
                splits_on.append('symbols')
        
        return {
            'splits_on': splits_on,
            'preserves': preserves
        }
    
    def _determine_primary_strategy(self, analysis: Dict) -> str:
        """Determine the primary pretokenization strategy"""
        splits_on = analysis.get('splits_on', [])
        
        if not splits_on:
            return 'character_level_or_whole_text'
        elif 'spaces' in splits_on and 'punctuation' in splits_on:
            return 'whitespace_and_punctuation'
        elif 'spaces' in splits_on:
            return 'whitespace_based'
        elif 'punctuation' in splits_on:
            return 'punctuation_based'
        else:
            return 'pattern_based'
    
    def _analyze_boundary_handling(self, examples: Dict) -> Dict[str, Any]:
        """Analyze how boundaries between different content types are handled"""
        boundary_analysis = {}
        
        # Word-number boundaries
        if 'numbers_mixed' in examples and 'tokens' in examples['numbers_mixed']:
            tokens = examples['numbers_mixed']['tokens']
            boundary_analysis['word_number'] = {
                'tokens': tokens,
                'preserves_boundary': 'test123' not in ' '.join(tokens)
            }
        
        # Case boundaries
        if 'camelCase' in examples and 'tokens' in examples['camelCase']:
            tokens = examples['camelCase']['tokens']
            boundary_analysis['case_changes'] = {
                'tokens': tokens,
                'splits_on_case': len(tokens) > len([w for w in ['camelCase', 'PascalCase']])
            }
        
        # Language boundaries
        multilingual_examples = ['japanese_mixed', 'korean_mixed']
        for example in multilingual_examples:
            if example in examples and 'tokens' in examples[example]:
                tokens = examples[example]['tokens']
                boundary_analysis[f'{example}_script_boundary'] = {
                    'tokens': tokens,
                    'token_count': len(tokens)
                }
        
        return boundary_analysis
    
    def _analyze_cross_linguistic_behavior(self, examples: Dict) -> Dict[str, Any]:
        """Analyze behavior across different languages and scripts"""
        linguistic_analysis = {}
        
        # Script families
        script_families = {
            'latin': ['mixed_case', 'contractions'],
            'cjk': ['chinese_no_spaces', 'japanese_mixed', 'korean_mixed'],
            'arabic': ['arabic_rtl', 'persian_farsi'],
            'indic': ['hindi_devanagari', 'thai_no_spaces'],
            'turkic': ['turkish_special']
        }
        
        for script_family, test_names in script_families.items():
            family_results = {}
            for test_name in test_names:
                if test_name in examples and 'tokens' in examples[test_name]:
                    example_data = examples[test_name]
                    family_results[test_name] = {
                        'token_count': example_data['token_count'],
                        'efficiency': example_data['char_to_token_ratio'],
                        'tokens': example_data['tokens'][:5]  # First 5 tokens for inspection
                    }
            linguistic_analysis[script_family] = family_results
        
        return linguistic_analysis
    
    def _analyze_normalization_effects(self, tokenizer) -> Dict[str, Any]:
        """Analyze Unicode normalization and other preprocessing effects"""
        import unicodedata
        
        normalization_tests = {
            'unicode_forms': {
                'nfc_composed': 'café naïve résumé',  # NFC form (composed)
                'nfd_decomposed': 'cafe\u0301 nai\u0308ve re\u0301sume\u0301',  # NFD form (decomposed)
                'nfkc_compat': 'ﬁle',  # NFKC compatibility (fi ligature)
                'nfkd_compat_decomp': unicodedata.normalize('NFKD', 'ﬁle')  # NFKD
            },
            'case_folding': {
                'mixed_case': 'Hello WORLD Café',
                'lowercase': 'hello world café',
                'uppercase': 'HELLO WORLD CAFÉ',
                'title_case': 'Hello World Café'
            },
            'whitespace_variants': {
                'normal_space': 'hello world test',
                'non_breaking_space': 'hello\u00a0world\u00a0test',
                'em_space': 'hello\u2003world\u2003test',
                'thin_space': 'hello\u2009world\u2009test',
                'zero_width_space': 'hello\u200bworld\u200btest',
                'tab_newline': 'hello\tworld\ntest'
            },
            'diacritics_comprehensive': {
                'latin_diacritics': 'àáâãäåæçèéêëìíîïñòóôõöøùúûüý',
                'combining_diacritics': 'a\u0300a\u0301a\u0302a\u0303a\u0308',  # grave, acute, circumflex, tilde, diaeresis
                'double_diacritics': 'ǘ',  # u with diaeresis and macron
                'mixed_scripts_diacritics': 'café мёд عَرَبِيّ',  # Latin, Cyrillic, Arabic with diacritics
                'vietnamese_complex': 'tiếng việt',  # Complex Vietnamese diacritics
                'turkish_special': 'İstanbul Güneş',  # Turkish i/I distinction
            },
            'contractions_detailed': {
                'apostrophe_contractions': "I'm you're they've we'll can't don't won't",
                'unicode_apostrophe': "I'm you're they've we'll can't don't won't",  # U+2019 right single quotation
                'multiple_apostrophes': "rock'n'roll '90s wi-fi's",
                'possessives': "John's car the dogs' toys children's books",
                'language_contractions': "c'est n'est l'eau d'accord",  # French
                'elision': "dell'arte all'italiana",  # Italian
            },
            'special_unicode_cases': {
                'emoji_sequences': '👨‍👩‍👧‍👦 🏳️‍🌈 👍🏽',  # Complex emoji with ZWJ and modifiers
                'rtl_marks': 'Hello \u202eמלש\u202d World',  # RTL override marks
                'bidi_text': 'Hello العالم مرحبا World',  # Mixed LTR/RTL
                'mathematical': '∑∏∆∇∞±√∫∂',
                'currency_symbols': '$€¥£₹₽₿',
                'script_mixing': 'Hello世界مرحباनमस्ते',  # Latin+Chinese+Arabic+Devanagari
            },
            'ligatures_and_presentation': {
                'latin_ligatures': 'ﬁle ﬂower ﬀ',  # fi, fl, ff ligatures
                'arabic_presentation': 'سلام',  # Arabic with different presentation forms
                'fullwidth_chars': 'Ｈｅｌｌｏ　Ｗｏｒｌｄ',  # Fullwidth Latin
                'subscript_superscript': 'H₂O E=mc² x²+y²=z²',
            }
        }
        
        results = {}
        for test_category, test_variants in normalization_tests.items():
            category_results = {}
            
            for variant_name, text in test_variants.items():
                try:
                    tokens = tokenizer.encode(text)
                    decoded = tokenizer.decode(tokens)
                    
                    # Detailed analysis
                    analysis = {
                        'original_length': len(text),
                        'decoded_length': len(decoded),
                        'token_count': len(tokens),
                        'perfect_reconstruction': text == decoded,
                        'length_preserved': len(text) == len(decoded),
                        'tokens_per_char': len(tokens) / len(text) if len(text) > 0 else 0
                    }
                    
                    # Character-level analysis
                    if text != decoded:
                        analysis['character_changes'] = self._analyze_character_changes(text, decoded)
                    
                    # Unicode normalization detection
                    analysis['unicode_analysis'] = self._analyze_unicode_changes(text, decoded)
                    
                    category_results[variant_name] = analysis
                    
                except Exception as e:
                    category_results[variant_name] = {'error': str(e)}
            
            # Cross-variant analysis
            if len(category_results) > 1:
                category_results['normalization_summary'] = self._summarize_normalization_behavior(category_results, test_category)
            
            results[test_category] = category_results
        
        return results
    
    def _analyze_character_changes(self, original: str, decoded: str) -> Dict[str, Any]:
        """Analyze what character-level changes occurred"""
        import difflib
        
        changes = {
            'insertions': [],
            'deletions': [],
            'substitutions': [],
            'position_shifts': []
        }
        
        # Use difflib for detailed character diff
        matcher = difflib.SequenceMatcher(None, original, decoded)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'insert':
                changes['insertions'].append({
                    'position': j1,
                    'text': decoded[j1:j2],
                    'context': decoded[max(0, j1-3):j2+3]
                })
            elif tag == 'delete':
                changes['deletions'].append({
                    'position': i1,
                    'text': original[i1:i2],
                    'context': original[max(0, i1-3):i2+3]
                })
            elif tag == 'replace':
                changes['substitutions'].append({
                    'original': original[i1:i2],
                    'decoded': decoded[j1:j2],
                    'position': i1
                })
        
        return changes
    
    def _analyze_unicode_changes(self, original: str, decoded: str) -> Dict[str, Any]:
        """Analyze Unicode-specific changes"""
        import unicodedata
        
        unicode_analysis = {
            'normalization_applied': False,
            'form_change': None,
            'combining_marks_changed': False,
            'script_changes': [],
            'category_changes': []
        }
        
        # Check for Unicode normalization
        for form in ['NFC', 'NFD', 'NFKC', 'NFKD']:
            normalized = unicodedata.normalize(form, original)
            if decoded == normalized and original != normalized:
                unicode_analysis['normalization_applied'] = True
                unicode_analysis['form_change'] = form
                break
        
        # Analyze combining marks
        orig_combining = len([c for c in original if unicodedata.combining(c)])
        decoded_combining = len([c for c in decoded if unicodedata.combining(c)])
        if orig_combining != decoded_combining:
            unicode_analysis['combining_marks_changed'] = True
            unicode_analysis['combining_change'] = {
                'original_count': orig_combining,
                'decoded_count': decoded_combining
            }
        
        # Script analysis
        def get_scripts(text):
            scripts = set()
            for char in text:
                try:
                    script = unicodedata.name(char).split()[0] if unicodedata.name(char, '') else 'UNKNOWN'
                    scripts.add(script)
                except:
                    scripts.add('UNKNOWN')
            return scripts
        
        orig_scripts = get_scripts(original)
        decoded_scripts = get_scripts(decoded)
        if orig_scripts != decoded_scripts:
            unicode_analysis['script_changes'] = {
                'original': list(orig_scripts),
                'decoded': list(decoded_scripts),
                'added': list(decoded_scripts - orig_scripts),
                'removed': list(orig_scripts - decoded_scripts)
            }
        
        return unicode_analysis
    
    def _summarize_normalization_behavior(self, results: Dict, category: str) -> Dict[str, Any]:
        """Summarize normalization behavior across variants"""
        summary = {
            'consistent_behavior': True,
            'normalization_type': 'none',
            'preservation_rate': 0.0,
            'common_changes': [],
            'insights': []
        }
        
        valid_results = {k: v for k, v in results.items() if isinstance(v, dict) and 'error' not in v}
        if not valid_results:
            return summary
        
        # Check consistency
        perfect_reconstructions = [r.get('perfect_reconstruction', False) for r in valid_results.values()]
        summary['preservation_rate'] = sum(perfect_reconstructions) / len(perfect_reconstructions)
        summary['consistent_behavior'] = len(set(perfect_reconstructions)) <= 1
        
        # Analyze normalization patterns
        if category == 'unicode_forms':
            nfc_result = results.get('nfc_composed', {})
            nfd_result = results.get('nfd_decomposed', {})
            
            if (nfc_result.get('perfect_reconstruction') and 
                nfd_result.get('perfect_reconstruction') and
                nfc_result.get('token_count') == nfd_result.get('token_count')):
                summary['normalization_type'] = 'form_preserving'
                summary['insights'].append('Preserves Unicode normalization forms')
            elif (nfc_result.get('perfect_reconstruction') and 
                  not nfd_result.get('perfect_reconstruction')):
                summary['normalization_type'] = 'nfc_normalizing'
                summary['insights'].append('Normalizes to NFC form')
            elif (not nfc_result.get('perfect_reconstruction') and 
                  nfd_result.get('perfect_reconstruction')):
                summary['normalization_type'] = 'nfd_normalizing'
                summary['insights'].append('Normalizes to NFD form')
        
        elif category == 'contractions_detailed':
            contraction_results = [r for k, r in valid_results.items() if 'contraction' in k]
            if contraction_results:
                perfect_rate = sum(r.get('perfect_reconstruction', False) for r in contraction_results) / len(contraction_results)
                if perfect_rate > 0.8:
                    summary['insights'].append('Preserves contractions well')
                elif perfect_rate < 0.2:
                    summary['insights'].append('Splits contractions aggressively')
                else:
                    summary['insights'].append('Mixed contraction handling')
        
        elif category == 'diacritics_comprehensive':
            diacritic_preservation = [r.get('perfect_reconstruction', False) for r in valid_results.values()]
            if all(diacritic_preservation):
                summary['insights'].append('Perfect diacritic preservation')
            elif not any(diacritic_preservation):
                summary['insights'].append('Strips or modifies diacritics')
            else:
                summary['insights'].append('Selective diacritic handling')
        
        return summary

    def run_analysis(self):
        """Run complete analysis on all tokenizers"""
        print("Starting comprehensive tokenizer analysis...")
        
        for tokenizer_name in self.tokenizer_configs.keys():
            tokenizer = self.load_tokenizer(tokenizer_name)
            
            if tokenizer is None:
                continue
                
            try:
                result = self.analyze_tokenizer(tokenizer_name, tokenizer)
                self.results.append(result)
                print(f"✓ Completed analysis for {tokenizer_name}")
            except Exception as e:
                print(f"✗ Failed to analyze {tokenizer_name}: {str(e)}")
                
        self.save_results()
        self.create_summary_table()

    def save_results(self):
        """Save detailed results to JSON"""
        for result in self.results:
            name = result["tokenizer_name"]
            with open(f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"Results saved to {name}.json")

        # with open('tokenizer_analysis_detailed.json', 'w', encoding='utf-8') as f:
        #     json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nDetailed results saved to: tokenizer_analysis_detailed.json")

    def create_summary_table(self):
        """Create summary CSV table"""
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'Tokenizer name': result['tokenizer_name'],
                'inherits from': result['inherits_from'],
                'diff from parent': result['diff_from_parent'],
                'Tokenization Method': result['tokenization_method'],
                'Vocab Size': result['vocab_size'],
                'Special Characters': result['special_characters'],
                'Byte-fallback': result['byte_fallback'],
                'Languages': result['script_coverage'],
                'Multi-lingual': result['multi_lingual'],
                'Numbers': result['numbers'],
                'Pretokenization Splits': result['pretokenization_splits'],
                'Unicode Normalization': result['unicode_normalization'],
                'Continuation': result['continuation'],
                'OOV Handling': result['oov_handling'],
                'Script Coverage': result['script_coverage'],
                'Whitespace Treatment': result['whitespace_treatment'],
                'Release Date': result['release_date']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv('tokenizer_comparison_table.csv', index=False)
        print(f"Summary table saved to: tokenizer_comparison_table.csv")
        
        # Create detailed pretokenization analysis
        self.create_pretokenization_report()
        
        # Print summary to console
        print("\n" + "="*80)
        print("TOKENIZER ANALYSIS SUMMARY")
        print("="*80)
        
        for result in self.results:
            print(f"\n{result['tokenizer_name']}:")
            print(f"  Method: {result['tokenization_method']}")
            print(f"  Vocab: {result['vocab_size']:,} tokens")
            print(f"  Multilingual: {result['multi_lingual']}")
            print(f"  Scripts: {result['script_coverage']}")
            
            # Pretokenization insights
            pretok_detail = result.get('pretokenization_detailed', {})
            if pretok_detail:
                strategy = pretok_detail.get('primary_strategy', 'unknown')
                splits = pretok_detail.get('splits_on', [])
                print(f"  Pretokenization: {strategy}")
                if splits:
                    print(f"    Splits on: {', '.join(splits)}")
            
            # Language-specific insights
            lang_analysis = result.get('language_analysis', {})
            print(f"  Language efficiency (tokens/char):")
            for lang, data in lang_analysis.items():
                if 'text' in data and 'error' not in data['text']:
                    ratio = data['text']['tokens_per_char']
            # Add detailed normalization insights to summary
        detailed_norm = result.get('pretokenization_detailed', {}).get('normalization_effects', {})
        norm_insights = []
        
        for category, category_data in detailed_norm.items():
            if isinstance(category_data, dict) and 'normalization_summary' in category_data:
                summary = category_data['normalization_summary']
                insights = summary.get('insights', [])
                if insights:
                    norm_insights.extend([f"{category}: {insight}" for insight in insights])
        
        if norm_insights:
            print(f"  Normalization insights: {'; '.join(norm_insights[:3])}")  # Show top 3

    def create_pretokenization_report(self):
        """Create a detailed pretokenization analysis report"""
        report_data = []
        
        for result in self.results:
            tokenizer_name = result['tokenizer_name']
            pretok_data = result.get('pretokenization_detailed', {})
            
            if not pretok_data or 'error' in pretok_data:
                continue
                
            # Extract key insights
            strategy = pretok_data.get('primary_strategy', 'unknown')
            splits_on = ', '.join(pretok_data.get('splits_on', []))
            preserves = ', '.join(pretok_data.get('preserves', []))
            
            # Language handling efficiency
            cross_ling = pretok_data.get('cross_linguistic_behavior', {})
            lang_efficiency = {}
            for script_family, family_data in cross_ling.items():
                if family_data:
                    avg_efficiency = sum(data.get('efficiency', 0) for data in family_data.values() if isinstance(data, dict)) / len(family_data)
                    lang_efficiency[script_family] = avg_efficiency
            
            # Boundary handling
            boundaries = pretok_data.get('boundary_handling', {})
            boundary_summary = []
            for boundary_type, boundary_data in boundaries.items():
                if isinstance(boundary_data, dict) and 'preserves_boundary' in boundary_data:
                    status = "preserves" if boundary_data['preserves_boundary'] else "merges"
                    boundary_summary.append(f"{boundary_type}:{status}")
            
            # Normalization behavior
            normalization = pretok_data.get('normalization_effects', {})
            norm_summary = []
            for norm_type, norm_data in normalization.items():
                if isinstance(norm_data, dict) and 'normalized' in norm_data:
                    status = "normalizes" if norm_data['normalized'] else "preserves"
                    norm_summary.append(f"{norm_type}:{status}")
            
            report_data.append({
                'tokenizer_name': tokenizer_name,
                'primary_strategy': strategy,
                'splits_on': splits_on,
                'preserves': preserves,
                'boundary_handling': '; '.join(boundary_summary),
                'normalization': '; '.join(norm_summary),
                'latin_efficiency': lang_efficiency.get('latin', 0),
                'cjk_efficiency': lang_efficiency.get('cjk', 0),
                'arabic_efficiency': lang_efficiency.get('arabic', 0),
                'indic_efficiency': lang_efficiency.get('indic', 0),
                'turkic_efficiency': lang_efficiency.get('turkic', 0)
            })
        
        # Save to CSV
        if report_data:
            df = pd.DataFrame(report_data)
            df.to_csv('pretokenization_detailed_analysis.csv', index=False)
            print(f"Detailed pretokenization analysis saved to: pretokenization_detailed_analysis.csv")
            
            # Print pretokenization insights
            print("\n" + "="*80)
            print("PRETOKENIZATION ANALYSIS INSIGHTS")
            print("="*80)
            
            for row in report_data:
                print(f"\n{row['tokenizer_name']}:")
                print(f"  Strategy: {row['primary_strategy']}")
                print(f"  Splits on: {row['splits_on']}")
                print(f"  Preserves: {row['preserves']}")
                print(f"  Boundary handling: {row['boundary_handling']}")
                print(f"  Cross-script efficiency:")
                print(f"    Latin: {row['latin_efficiency']:.2f}")
                print(f"    CJK: {row['cjk_efficiency']:.2f}")
                print(f"    Arabic: {row['arabic_efficiency']:.2f}")
                
        # Save detailed examples for manual inspection
        # self.save_pretokenization_examples()

if __name__ == "__main__":
    analyzer = TokenizerAnalyzer()
    analyzer.run_analysis()