# automatic_singer_songwriter_project
## Outline
全自動で
1. 作詞
1. 作曲
1. 編曲
1. 歌唱
するシステムの構築を目指します。

## run
1. 'docker-compose up'
1. 'curl -X POST -H "Content-Type: application/json" -d \
'{"seed_text":"交通量調査", "beam_depth":3}'\
 http://localhost:5000/predict
' 
	1. seed_text:生成の元になる文章。これの続きを生成する。
	1. beam_depth:生成する深さ。続きを何単語予測するか

1. 
1. 
1. 


## 作詞 lyric generation
- [ ] 文書生成についてのサーベイ
- [ ] 公開されているモデルのテスト
- [ ] 日本語モデルの実装


## 作曲
- [ ] 自動作曲についてのサーベイ
- [ ] 公開されているモデルのテスト
- [ ] ジャンルなどのパラメータを指定できるモデル設計


## 編曲
- [ ] 自動編曲についてのサーベイ
- [ ] 公開されているモデルのテスト
- [ ] ジャンルなどのパラメータを指定できるモデル設計


## 歌唱
- [ ] Text To Speech、Vocoder、Voice Convertionについてのサーベイ
- [ ] 公開されているモデルのテスト
- [ ] 日本語のモデル設計、実装


