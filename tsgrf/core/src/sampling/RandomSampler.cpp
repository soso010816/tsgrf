/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).
  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#include <algorithm>
#include <random>
#include <cstddef>

#include "RandomSampler.h"

namespace grf {

RandomSampler::RandomSampler(uint seed,
                             const SamplingOptions& options) :
    options(options) {
  random_number_generator.seed(seed);
}

void RandomSampler::sample_clusters(size_t num_rows,
                                    double sample_fraction,
                                    std::vector<size_t>& samples) {

  if (options.get_clusters().empty()) {
    sample(num_rows, sample_fraction, samples);
  } else {
    // num_samples 为集群种类的数目
    size_t num_samples = options.get_clusters().size();
    sample(num_samples, sample_fraction, samples);
  }
}

// 重写sample_clusters函数，增加了blocks参数
void RandomSampler::sample_clusters(size_t num_rows,
                                    double sample_fraction,
                                    std::vector<size_t>& samples,
                                    std::vector<std::vector<size_t>>& blocks,
                                    int block_group_size) {

  if (options.get_clusters().empty()) {
    sample(num_rows, sample_fraction, samples, blocks, block_group_size);
  } else {
    // num_samples 为集群种类的数目
    size_t num_samples = options.get_clusters().size();
    sample(num_samples, sample_fraction, samples, blocks, block_group_size);
  }
}

void RandomSampler::sample(size_t num_samples,
                           double sample_fraction,
                           std::vector<size_t>& samples) {
  // static_cast 表示类型的转化，由浮点数转化为序列数
  size_t num_samples_inbag = static_cast<size_t>(num_samples * sample_fraction);
  shuffle_and_split(samples, num_samples, num_samples_inbag);
}

// ------------------------------------------------------------------------------
void RandomSampler::sample(size_t num_samples,
                           double sample_fraction,
                           std::vector<size_t>& samples,
                           std::vector<std::vector<size_t>>& blocks,
                           int block_group_size) {

  // static_cast 表示类型的转化，由浮点数转化为序列数
  size_t num_samples_inbag = static_cast<size_t>(num_samples * sample_fraction);
  // -------------------add method: block split-------------------------
  block_and_split(samples, num_samples, sample_fraction, blocks, block_group_size);
}
// ------------------------------------------------------------------------------

/* -------------新增函数---------------------------------------------
  写这个函数的主要目的是，应对 ci.group.size>1,
  作用是对sample 后的 blocks 再次按sample.fraction*2 进行二次block抽样
  */
void RandomSampler::subsample_for_cigroup(const std::vector<size_t>& samples,
                                          const std::vector<std::vector<size_t>>& blocks,
                                          double sample_fraction,
                                          std::vector<size_t>& subsamples,
                                          std::vector<std::vector<size_t>>& blocks_subsamples){
  std::vector<std::vector<size_t>> shuffled_blocks(blocks);
  nonstd::shuffle(shuffled_blocks.begin(), shuffled_blocks.end(), random_number_generator);

  size_t block_subsample_size = static_cast<size_t>(std::round(shuffled_blocks.size() * sample_fraction));
  shuffled_blocks.resize(block_subsample_size);

  std::sort(shuffled_blocks.begin(), shuffled_blocks.end(), [&](const std::vector<size_t>& a, const std::vector<size_t>& b) {
      return a.front() < b.front();
  });

  blocks_subsamples = shuffled_blocks;

  subsamples.clear();
  for (const auto& block : shuffled_blocks) {
      for (size_t element : block) {
          subsamples.push_back(element);
      }
  }
}
// ---------------------------------------------------------------

void RandomSampler::subsample(const std::vector<size_t>& samples,
                              double sample_fraction,
                              std::vector<size_t>& subsamples) {
  // 创建一个新的动态数组,复杂并打乱原来的 samples
  std::vector<size_t> shuffled_sample(samples);
  nonstd::shuffle(shuffled_sample.begin(), shuffled_sample.end(), random_number_generator);

  uint subsample_size = (uint) std::ceil(samples.size() * sample_fraction); 
  subsamples.resize(subsample_size);
  std::copy(shuffled_sample.begin(),
            shuffled_sample.begin() + subsamples.size(),
            subsamples.begin());
}

/* 
oob_subsample 将未被抽中的样本（袋外样本）存储在 oob_samples 中。
将样本打乱并分割为子样本和 oob 样本（用于honesty tree中），
注意 honest dataset是在 sample.fraction 基础上进行再抽样.
*/
void RandomSampler::subsample(const std::vector<size_t>& samples,
                              double sample_fraction,
                              std::vector<size_t>& subsamples,
                              std::vector<size_t>& oob_samples) {
  // 注意上述传来的 subsamples 和 oob_samples 是空的动态数组
  std::vector<size_t> shuffled_sample(samples);
  nonstd::shuffle(shuffled_sample.begin(), shuffled_sample.end(), random_number_generator);

  size_t subsample_size = (size_t) std::ceil(samples.size() * sample_fraction);
  subsamples.resize(subsample_size);
  oob_samples.resize(samples.size() - subsample_size);

  std::copy(shuffled_sample.begin(),
            shuffled_sample.begin() + subsamples.size(),
            subsamples.begin());
  std::copy(shuffled_sample.begin() + subsamples.size(),
            shuffled_sample.end(),
            oob_samples.begin());
}

/* ------------------重写存在诚实树时的 抽样方法 subsample:
  抽样方法为: 
  0：blocks完全随机抽样 [3,2],[1,4]，
   1：一前一后逐个抽样 [1,3],[2,4]，
   2：半段半段抽样: [1,2,3,4]->[1,2],[3,4]，
   3: block 内固定时间窗口抽样: [1,2,3,4]->[2,3],[1,4]]
   4: block 内随机抽样
------------------------ */
void RandomSampler::subsample(const std::vector<size_t>& samples,
                              const std::vector<std::vector<size_t>>& blocks,
                              const TreeOptions& options,
                              std::vector<size_t>& subsamples,
                              std::vector<size_t>& oob_samples) {

    double sample_fraction = options.get_honesty_fraction();
    size_t honesty_method = options.get_honesty_method();
    
    if(honesty_method == 0){
      // 0：直接随机打乱所有样本，进行抽样
        subsample_sub0(samples, blocks, sample_fraction, subsamples, oob_samples);
    }else if(honesty_method == 1){
      // 1: 对block 内样本，按一前一后的顺序进行抽样
        subsample_sub1(samples, blocks, sample_fraction, subsamples, oob_samples);
    }else if(honesty_method == 2){
      // 2: 对 block 内样本，按前一半训练，后一半预测的顺序进行抽样
        subsample_sub2(samples, blocks, sample_fraction, subsamples, oob_samples);
    }else if(honesty_method == 3){
      // 3: 对 block 内样本，按连续时间窗口进行抽样
        subsample_sub3(samples, blocks, sample_fraction, subsamples, oob_samples);
    }else{
      // 4：对 block 内样本，随机抽样
        subsample_sub4(samples, blocks, sample_fraction, subsamples, oob_samples);
    }
}
//---------------------------------重现诚实树抽样的方法选择 subsample_sub-----------------------------------------------
void RandomSampler::subsample_sub0(const std::vector<size_t>& samples,
                                  const std::vector<std::vector<size_t>>& blocks,
                                  double sample_fraction,
                                  std::vector<size_t>& subsamples,
                                  std::vector<size_t>& oob_samples) {
  std::vector<size_t> shuffled_sample(samples);
  std::vector<std::vector<size_t>> shuffled_blocks(blocks);

  // shuffle the sample
  nonstd::shuffle(shuffled_sample.begin(), shuffled_sample.end(), random_number_generator);
  // nonstd::shuffle(shuffled_blocks.begin(), shuffled_blocks.end(), random_number_generator);

  size_t subsample_size = (size_t) std::ceil(samples.size() * sample_fraction);
  subsamples.resize(subsample_size);
  oob_samples.resize(samples.size() - subsample_size);

  std::copy(shuffled_sample.begin(),
            shuffled_sample.begin() + subsamples.size(),
            subsamples.begin());
  std::copy(shuffled_sample.begin() + subsamples.size(),
            shuffled_sample.end(),
            oob_samples.begin());
}

void RandomSampler::subsample_sub1(const std::vector<size_t>& samples,
                                   const std::vector<std::vector<size_t>>& blocks,
                                   double sample_fraction,
                                   std::vector<size_t>& subsamples,
                                   std::vector<size_t>& oob_samples) {
    for (const auto& block : blocks) {
        size_t total_samples = block.size();
        size_t subsample_size = static_cast<size_t>(std::ceil(total_samples * sample_fraction));
        
        // 确定subsample和oob_samples的选择策略
        if (subsample_size <= total_samples / 2) {
            // 当需要的subsample数量小于等于一半时，交替选择subsample和oob_samples
            for (size_t idx = 0; idx < total_samples; ++idx) {
                if (idx < 2 * subsample_size) {
                    if (idx % 2 == 0) {
                        subsamples.push_back(block[idx]);
                    } else {
                        oob_samples.push_back(block[idx]);
                    }
                } else {
                    // 剩余的所有样本作为oob
                    oob_samples.push_back(block[idx]);
                }
            }
        } else {
            // 当需要的subsample数量大于一半时
            // 先从block的前部分抽取多余数目的样本作为subsample
            size_t extra_subsamples = subsample_size - total_samples / 2;
            for (size_t idx = 0; idx < extra_subsamples; ++idx) {
                subsamples.push_back(block[idx]);
            }
            // 然后交替选择subsample和oob_samples
            for (size_t idx = extra_subsamples; idx < total_samples; ++idx) {
                if ((idx - extra_subsamples) % 2 == 0) {
                    subsamples.push_back(block[idx]);
                } else {
                    oob_samples.push_back(block[idx]);
                }
            }
        }
    }
}



void RandomSampler::subsample_sub2(const std::vector<size_t>& samples,
                                  const std::vector<std::vector<size_t>>& blocks,
                                  double sample_fraction,
                                  std::vector<size_t>& subsamples,
                                  std::vector<size_t>& oob_samples) {
    // 遍历每个block
    for (const auto& block : blocks) {
        // 计算每个block中要选取的样本数
        size_t block_subsample_size = static_cast<size_t>(std::ceil(block.size() * sample_fraction));
        subsamples.insert(subsamples.end(), block.begin(), block.begin() + block_subsample_size);
        if (block_subsample_size < block.size()) {
            oob_samples.insert(oob_samples.end(), block.begin() + block_subsample_size, block.end());
        }
    }
}

void RandomSampler::subsample_sub3(const std::vector<size_t>& samples,
                                  const std::vector<std::vector<size_t>>& blocks,
                                  double sample_fraction,
                                  std::vector<size_t>& subsamples,
                                  std::vector<size_t>& oob_samples) {

    size_t window_size = static_cast<size_t>(std::ceil(blocks[0].size() * sample_fraction));

    for (const auto& block : blocks) {

        if (window_size >= block.size()) {
            subsamples.insert(subsamples.end(), block.begin(), block.end());
        } else {
            size_t start_index = rand() % (block.size() - window_size + 1);

            subsamples.insert(subsamples.end(), block.begin() + start_index, block.begin() + start_index + window_size);

            if (start_index > 0) {
                oob_samples.insert(oob_samples.end(), block.begin(), block.begin() + start_index);
            }
            if (start_index + window_size < block.size()) {
                oob_samples.insert(oob_samples.end(), block.begin() + start_index + window_size, block.end());
            }
        }
    }
}

void RandomSampler::subsample_sub4(const std::vector<size_t>& samples,
                                  const std::vector<std::vector<size_t>>& blocks,
                                  double sample_fraction,
                                  std::vector<size_t>& subsamples,
                                  std::vector<size_t>& oob_samples) {

    size_t block_subsample_size = static_cast<size_t>(std::ceil(blocks[0].size() * sample_fraction));
    for (const auto& block : blocks) {
        std::vector<size_t> shuffled_block(block);
        nonstd::shuffle(shuffled_block.begin(), shuffled_block.end(), random_number_generator);
        subsamples.insert(subsamples.end(), shuffled_block.begin(), shuffled_block.begin() + block_subsample_size);
        if (block_subsample_size < block.size()) {
            oob_samples.insert(oob_samples.end(), block.begin() + block_subsample_size, block.end());
        }
    }                                
}
// ------------------------------------------------------------------------------


// 按指定 size 进行子样本抽样
void RandomSampler::subsample_with_size(const std::vector<size_t>& samples,
                                        size_t subsample_size,
                                        std::vector<size_t>& subsamples) {
  // 复制 sample 并转至 shuffled_sample
  std::vector<size_t> shuffled_sample(samples);
  nonstd::shuffle(shuffled_sample.begin(), shuffled_sample.end(), random_number_generator);
  // 重置 subsamples 的大小
  subsamples.resize(subsample_size);
  /* 将 shuffled_sample 容器中的一部分元素（从 begin() 到 begin() + subsamples.size()）
  复制到 subsamples 容器中，从 subsamples.begin() 开始的位置开始存储。
   */
  std::copy(shuffled_sample.begin(),
            shuffled_sample.begin() + subsamples.size(),
            subsamples.begin());
}

/*依据集群进行抽样，如果无指定集群则返回原样本 */
void RandomSampler::sample_from_clusters(const std::vector<size_t>& clusters,
                                         std::vector<size_t>& samples) {
  if (options.get_clusters().empty()) {
    samples = clusters;
  } else {
    const std::vector<std::vector<size_t>>& samples_by_cluster = options.get_clusters();
    for (size_t cluster : clusters) {
      const std::vector<size_t>& cluster_samples = samples_by_cluster[cluster];

      // Draw samples_per_cluster observations from each cluster. If the cluster is
      // smaller than the samples_per_cluster parameter, just use the whole cluster.
      if (cluster_samples.size() <= options.get_samples_per_cluster()) {
        samples.insert(samples.end(), cluster_samples.begin(), cluster_samples.end());
      } else {
        std::vector<size_t> subsamples;
        subsample_with_size(cluster_samples, options.get_samples_per_cluster(), subsamples);
        samples.insert(samples.end(), subsamples.begin(), subsamples.end());
      }
    }
  }
}


/* 没有收到 samples_per_cluster 时的抽样，直接将样本中每个集群整体纳入*/
void RandomSampler::get_samples_in_clusters(const std::vector<size_t>& clusters,
                                            std::vector<size_t>& samples) {
  if (options.get_clusters().empty()) {
    samples = clusters;
  } else {
    for (size_t cluster : clusters) {
      const std::vector<size_t>& cluster_samples = options.get_clusters()[cluster];
      samples.insert(samples.end(), cluster_samples.begin(), cluster_samples.end());
    }
  }
}

// 将指定样本打乱和分割抽样
void RandomSampler::shuffle_and_split(std::vector<size_t>& samples,
                                      size_t n_all,
                                      size_t size) {
  // 因为 samples 是空的动态数组，所以 resize 为 n_all                                      
  samples.resize(n_all);

  // 填充动态数组，从 0 到 n_all-1，表示样本索引
  std::iota(samples.begin(), samples.end(), 0);
  // 使用随机数生成器随机打乱
  nonstd::shuffle(samples.begin(), samples.end(), random_number_generator);
  // resize 获取前 size 个元素
  samples.resize(size);
}

//----------------block split------------------------
void RandomSampler::block_and_split(std::vector<size_t>& samples,
                                    size_t n_all,
                                    double sample_fraction,
                                    std::vector<std::vector<size_t>>& blocks,
                                    int block_group_size) { 
  
  size_t block_num = (size_t) std::ceil(std::pow(n_all, 1.0 / block_group_size));
  size_t block_size = (size_t) std::floor(n_all / block_num);
  size_t block_sample_num  =(size_t) std::ceil(block_size * sample_fraction);
  samples.resize(block_sample_num * block_size);
  size_t index = 0;
  for (size_t i = 0; i < block_sample_num; i++){
    size_t start_index = rand() % (n_all - block_size + 1);
    std::vector<size_t> block(block_size);
    std::iota(block.begin(), block.end(), start_index);
    std::iota(samples.begin() + index, samples.begin() + index + block_size, start_index);
    blocks.push_back(block);
    index += block_size;
  }
}
//----------------------------------------------

void RandomSampler::draw(std::vector<size_t>& result,
                         size_t max,
                         const std::set<size_t>& skip,
                         size_t num_samples) {
  if (num_samples < max / 10) {
    draw_simple(result, max, skip, num_samples);
  } else {
    draw_fisher_yates(result, max, skip, num_samples);
  }
}

void RandomSampler::draw_simple(std::vector<size_t>& result,
                                size_t max,
                                const std::set<size_t>& skip,
                                size_t num_samples) {
  result.resize(num_samples);

  // Set all to not selected
  std::vector<bool> temp;
  temp.resize(max, false);

  nonstd::uniform_int_distribution<size_t> unif_dist(0, max - 1 - skip.size());
  for (size_t i = 0; i < num_samples; ++i) {
    size_t draw;
    do {
      draw = unif_dist(random_number_generator);
      for (auto& skip_value : skip) {
        if (draw >= skip_value) {
          ++draw;
        }
      }
    } while (temp[draw]);
    temp[draw] = true;
    result[i] = draw;
  }
}

void RandomSampler::draw_fisher_yates(std::vector<size_t>& result,
                                      size_t max,
                                      const std::set<size_t>& skip,
                                      size_t num_samples) {

  // Populate result vector with 0,...,max-1
  result.resize(max);
  std::iota(result.begin(), result.end(), 0);

  // Remove values that are to be skipped
  std::for_each(skip.rbegin(), skip.rend(),
                [&](size_t i) { result.erase(result.begin() + i); }
  );

  // Draw without replacement using Fisher Yates algorithm
  nonstd::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (size_t i = 0; i < num_samples; ++i) {
    size_t j = static_cast<size_t>(i + distribution(random_number_generator) * (max - skip.size() - i));
    std::swap(result[i], result[j]);
  }

  result.resize(num_samples);
}

size_t RandomSampler::sample_poisson(size_t mean) {
  nonstd::poisson_distribution<size_t> distribution(static_cast<double>(mean));
  return distribution(random_number_generator);
}

} // namespace grf
