## **0.5.0**&emsp;<sub><sup>2023-03-09 ([711ebb1...80acd75](https://github.com/yzx9/swcgeom/compare/711ebb1308d2a70eddcf6295a40883a7815e887e...80acd75f7a9c232008f5ce4c9ec1f43488efd21c?diff=split))</sup></sub>

### Features

##### `analysis`

- custom draw ([bdc1291](https://github.com/yzx9/swcgeom/commit/bdc1291eb4e213ad3918c70dad54f00d81be6769))
- re\-design feature_extractor ([e5affa7](https://github.com/yzx9/swcgeom/commit/e5affa700128d62e175bb6243ba1d7f00ac27c85))
- add populations feature extractor ([94a2bc6](https://github.com/yzx9/swcgeom/commit/94a2bc69a9eb1856138e54fc858a48480c324c61))
- plot histogram ([0dc05d7](https://github.com/yzx9/swcgeom/commit/0dc05d72a7b1dcff8f2e64e6b79a91e3d040ac8a))
- custom length plot ([d32272d](https://github.com/yzx9/swcgeom/commit/d32272d242274e8bdf2acd4d3dc80e5858481449))
- custom position of direction indicator ([b84b57e](https://github.com/yzx9/swcgeom/commit/b84b57eb912da55b54b4d72e526d4cf7e96abc64))
- support hidden swc in legend ([80acd75](https://github.com/yzx9/swcgeom/commit/80acd75f7a9c232008f5ce4c9ec1f43488efd21c))

##### `core`

- add populations ([711ebb1](https://github.com/yzx9/swcgeom/commit/711ebb1308d2a70eddcf6295a40883a7815e887e))
- rename class method option ([7003c1d](https://github.com/yzx9/swcgeom/commit/7003c1dc017a0ca432265c003434e871ea9c1fc6))
- take intersection among populations ([6442572](https://github.com/yzx9/swcgeom/commit/6442572c9fe32621dd93b6288d82e40a45e6012f))
- add pure\-func style utils ([7f9cba1](https://github.com/yzx9/swcgeom/commit/7f9cba14dd4bc0e018fe03bccd1cf845d4a3a421))
- add \`BranchTree\.from_eswc\` ([d01cd04](https://github.com/yzx9/swcgeom/commit/d01cd04505649cc782edbc34d5ac7205093b57de))
- add labels for populations ([998566e](https://github.com/yzx9/swcgeom/commit/998566e4c478716ad57a77b52dc6774608d4e797))

### Merges

- branch 'feature/populations' ([1c7d1e8](https://github.com/yzx9/swcgeom/commit/1c7d1e87a6a34ab79146adfe9e5e95b24f636432))

### BREAKING CHANGES

- `analysis` ([e5affa7](https://github.com/yzx9/swcgeom/commit/e5affa700128d62e175bb6243ba1d7f00ac27c85))
  - \`extract_feature\` has been fully re\-designed
  - \`NodeFeatures\.get_distribution\` has been removed
  - \`Sholl\.get_distribution\` has been removed
- `analysis` remove filters in \`NodeFeatures\` ([f0b37bf](https://github.com/yzx9/swcgeom/commit/f0b37bf075fef80a71442d75ae84132796f52c28))
- `core` \`Population.from_swc\` and \`Population\.from_eswc\` argument \`suffix\` has been renamed to \`ext\` ([7003c1d](https://github.com/yzx9/swcgeom/commit/7003c1dc017a0ca432265c003434e871ea9c1fc6))
- `core` add pure\-func style utils ([7f9cba1](https://github.com/yzx9/swcgeom/commit/7f9cba14dd4bc0e018fe03bccd1cf845d4a3a421))

## **0.4.1** <sub><sup>2023-03-01 ([4efbcef...4efbcef](https://github.com/yzx9/swcgeom/compare/4efbcef...4efbcef?diff=split))</sup></sub>

### Bug Fixes

##### `core`

- should count nonzero nodes ([4efbcef](https://github.com/yzx9/swcgeom/commit/4efbcef))

## **0.4.0** <sub><sup>2023-02-27 ([af51d11...1d9aeb5](https://github.com/yzx9/swcgeom/compare/af51d11...1d9aeb5?diff=split))</sup></sub>

### Features

##### `analysis`

- forward kwarg to plot method ([1d9aeb5](https://github.com/yzx9/swcgeom/commit/1d9aeb5))

##### `core`

- add \`Tree\.from_data_frame\` ([b36f382](https://github.com/yzx9/swcgeom/commit/b36f382))
- assemble lines into swc ([382df7b](https://github.com/yzx9/swcgeom/commit/382df7b))
- sort nodes after assemble ([d00687c](https://github.com/yzx9/swcgeom/commit/d00687c))
- rename \`sort_swc\` to \`sort_nodes\` ([b7f27d6](https://github.com/yzx9/swcgeom/commit/b7f27d6))
- support export swc without source ([a8da734](https://github.com/yzx9/swcgeom/commit/a8da734))

##### `transform`

- support cut short branch ([0bc2f2c](https://github.com/yzx9/swcgeom/commit/0bc2f2c))

### Performance Improvements

##### `core`

- reduce the number of value fetches ([33169d1](https://github.com/yzx9/swcgeom/commit/33169d1))

### BREAKING CHANGES

- `analysis` forward kwarg to plot method ([1d9aeb5](https://github.com/yzx9/swcgeom/commit/1d9aeb5))

## **0.3.1** <sub><sup>2023-02-13 ([557693b...fc9335f](https://github.com/yzx9/swcgeom/compare/557693b...fc9335f?diff=split))</sup></sub>

_no relevant changes_

## **0.3.0** <sub><sup>2023-02-12 ([40ff824...cedf52f](https://github.com/yzx9/swcgeom/compare/40ff824...cedf52f?diff=split))</sup></sub>

### Features

- add tree cutter ([e8f1e68](https://github.com/yzx9/swcgeom/commit/e8f1e68))
- support smart color swither ([a6df232](https://github.com/yzx9/swcgeom/commit/a6df232))
- add debug helper ([f3819a5](https://github.com/yzx9/swcgeom/commit/f3819a5))
- sort tree ([5f1d53e](https://github.com/yzx9/swcgeom/commit/5f1d53e))
- support to swc string ([0b68b65](https://github.com/yzx9/swcgeom/commit/0b68b65))
- sort swc tree ([d812c0d](https://github.com/yzx9/swcgeom/commit/d812c0d))

##### `analysis`

- add get distribution ([ea44851](https://github.com/yzx9/swcgeom/commit/ea44851))
- add population feature extractor ([7f3ff59](https://github.com/yzx9/swcgeom/commit/7f3ff59))
- plot feature distribution ([460f7d1](https://github.com/yzx9/swcgeom/commit/460f7d1))
- support plot sholl ([a253d60](https://github.com/yzx9/swcgeom/commit/a253d60))
- expose plot ax ([9bb7585](https://github.com/yzx9/swcgeom/commit/9bb7585))

##### `analysis/feature`

- expose ax ([2f98884](https://github.com/yzx9/swcgeom/commit/2f98884))
- minor ([bf18b4d](https://github.com/yzx9/swcgeom/commit/bf18b4d))
- change default options ([c6891b0](https://github.com/yzx9/swcgeom/commit/c6891b0))

##### `analysis/visualization`

- add legend ([3739d93](https://github.com/yzx9/swcgeom/commit/3739d93))

##### `core`

- add suffix option ([ad8b1b0](https://github.com/yzx9/swcgeom/commit/ad8b1b0))_ rename \`get_branch\` to \`branch\` ([7adca54](https://github.com/yzx9/swcgeom/commit/7adca54))_ export eswc ([0bce6a8](https://github.com/yzx9/swcgeom/commit/0bce6a8))

##### `core/swc`

- write 1 as root id by default ([946084b](https://github.com/yzx9/swcgeom/commit/946084b))
- support eswc ([f49906a](https://github.com/yzx9/swcgeom/commit/f49906a))
- fix multi roots ([de75ee2](https://github.com/yzx9/swcgeom/commit/de75ee2))
- sort tree ([a7ac4c7](https://github.com/yzx9/swcgeom/commit/a7ac4c7))
- rename read params ([cb55c2b](https://github.com/yzx9/swcgeom/commit/cb55c2b))

##### `core/tree`

- expose ndata id ([0464b3f](https://github.com/yzx9/swcgeom/commit/0464b3f))

##### `datasets`

- rename \`TreeToBranchTree\` to \`ToBranchTree\` ([5f6c93c](https://github.com/yzx9/swcgeom/commit/5f6c93c))

##### `transform`

- rename option ([ced31b1](https://github.com/yzx9/swcgeom/commit/ced31b1))
- add image stack ([114f119](https://github.com/yzx9/swcgeom/commit/114f119))
- add verbose option ([31a32a2](https://github.com/yzx9/swcgeom/commit/31a32a2))
- improve perf ([40c650d](https://github.com/yzx9/swcgeom/commit/40c650d))

##### `transform/image-stack`

- transforms population ([a686fb1](https://github.com/yzx9/swcgeom/commit/a686fb1))

##### `transform/image_stack`

- support big image stack ([bbe5bd1](https://github.com/yzx9/swcgeom/commit/bbe5bd1))
- add photometric ([35e40c5](https://github.com/yzx9/swcgeom/commit/35e40c5))

### Bug Fixes

- should copy ndata ([cedf52f](https://github.com/yzx9/swcgeom/commit/cedf52f))

##### `analysis/visulization`

- handle draw too many neurons in same axes ([de48fe2](https://github.com/yzx9/swcgeom/commit/de48fe2))

##### `core`

- avoid duplicate export tree ([4a506bb](https://github.com/yzx9/swcgeom/commit/4a506bb))
- tip should not be parent ([ce22ea0](https://github.com/yzx9/swcgeom/commit/ce22ea0))
- branch should be nodes ([3f1dcb0](https://github.com/yzx9/swcgeom/commit/3f1dcb0))
- should returns id ([c622666](https://github.com/yzx9/swcgeom/commit/c622666))
- handle zero length ([0112949](https://github.com/yzx9/swcgeom/commit/0112949))
- should propagate removal ([f87ab43](https://github.com/yzx9/swcgeom/commit/f87ab43))

##### `datasets/dgl`

- copy prop ([40ff824](https://github.com/yzx9/swcgeom/commit/40ff824))

##### `transfrom/image_stack`

- handle root sdf ([ce8cae2](https://github.com/yzx9/swcgeom/commit/ce8cae2))

### Performance Improvements

##### `transform/image_stack`

- boost intersect ([8003026](https://github.com/yzx9/swcgeom/commit/8003026))

##### `transforms`

- boost is_in of sdf ([45fee56](https://github.com/yzx9/swcgeom/commit/45fee56))

### BREAKING CHANGES

- `analysis` \`get_distribution\` returns both \`x\` and \`y\`, before ([460f7d1](https://github.com/yzx9/swcgeom/commit/460f7d1))<br>only \`y\`\.
- `anlysis/node_feature` change default options ([c6891b0](https://github.com/yzx9/swcgeom/commit/c6891b0))

## **0.2.0** <sub><sup>2022-09-28</sup></sub>

### BREAKING CHANGE

- several features have been activatedm, take advantage
- migrate to numpy-based, remove networkx

## **0.1.7** <sub><sup>2022-08-28</sup></sub>

### Features

##### `core`

- expose branch standardize method

##### `data/torch`

- support tree transfroms
- expose `BranchToTensor` transform, and support more options

##### `data/transforms`

- add branch transforms

### Bug Fixes

##### `core`

- stack arrays to get the correct shape
- update node correctly

##### `data/torch`

- super object is not subscriptable

### Performance Improvements

##### `data`

- delay log formatting

## **0.1.6** <sub><sup>2022-08-25</sup></sub>

### Features

- add `Branch.from_numpy`
- rename `Branch.from_numpy` to `Branch.from_numpy_batch`
- expose branch resampler

### Bug Fixes

##### `data/torch`

- avoid cycle reference

### BREAKING CHANGES

- resample args should be named arguments now
- rename `Branch.from_numpy` to `Branch.from_numpy_batch`, and drop the support
  for squeezed input

## **0.1.5** <sub><sup>2022-08-24</sup></sub>

### BREAKING CHANGE

- now node attributes are all dictionary attributes

### Features

- convert node to dict

##### `data/torch`

- add datasets

## **0.1.4** <sub><sup>2022-08-22</sup></sub>

### Bug Fixes

- import version correctly

## **0.1.3** <sub><sup>2022-08-21</sup></sub>

## **0.1.2** <sub><sup>2022-08-21</sup></sub>

## **0.1.1** <sub><sup>2022-08-21</sup></sub>

## **0.1.0** <sub><sup>2022-08-21</sup></sub>

### Features

- import codes
