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
#include <ctime>
#include <future>
#include <stdexcept>

#include "commons/utility.h"
#include "ForestTrainer.h"
#include "random/random.hpp"


namespace grf {

ForestTrainer::ForestTrainer(std::unique_ptr<RelabelingStrategy> relabeling_strategy,
                             std::unique_ptr<SplittingRuleFactory> splitting_rule_factory,
                             std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy) :
    tree_trainer(std::move(relabeling_strategy),
                 std::move(splitting_rule_factory),
                 std::move(prediction_strategy)) {}
    // ForestTrainer 包含了训练一个树的所有函数

Forest ForestTrainer::train(const Data& data, const ForestOptions& options) const {
  // 所有的树被存储在一个 std::vector 中，train_trees 将返回多颗树
  std::vector<std::unique_ptr<Tree>> trees = train_trees(data, options);

  size_t num_variables = data.get_num_cols() - data.get_disallowed_split_variables().size();
  size_t ci_group_size = options.get_ci_group_size();
  return Forest(trees, num_variables, ci_group_size);
}

std::vector<std::unique_ptr<Tree>> ForestTrainer::train_trees(const Data& data,
                                                              const ForestOptions& options) const {
  size_t num_samples = data.get_num_rows();
  uint num_trees = options.get_num_trees();

  // Ensure that the sample fraction is not too small and honesty fraction is not too extreme.
  const TreeOptions& tree_options = options.get_tree_options();
  bool honesty = tree_options.get_honesty();
  double honesty_fraction = tree_options.get_honesty_fraction();

  // 判断树的最小样本数目
  if ((size_t) num_samples * options.get_sample_fraction() < 1) {
    throw std::runtime_error("The sample fraction is too small, as no observations will be sampled.");
  } else if (honesty && ((size_t) num_samples * options.get_sample_fraction() * honesty_fraction < 1
             || (size_t) num_samples * options.get_sample_fraction() * (1-honesty_fraction) < 1)) {
    throw std::runtime_error("The honesty fraction is too close to 1 or 0, as no observations will be sampled.");
  }

  // 计算树将被分成的组数，每组包含由置信区间组大小指定的树的数量
  uint num_groups = static_cast<uint>(num_trees / options.get_ci_group_size());

  std::vector<uint> thread_ranges;
  split_sequence(thread_ranges, 0, num_groups - 1, options.get_num_threads());

  std::vector<std::future<std::vector<std::unique_ptr<Tree>>>> futures;
  futures.reserve(thread_ranges.size());

  std::vector<std::unique_ptr<Tree>> trees;
  trees.reserve(num_trees);

  for (uint i = 0; i < thread_ranges.size() - 1; ++i) {
    size_t start_index = thread_ranges[i];
    size_t num_trees_batch = thread_ranges[i + 1] - start_index;

    futures.push_back(std::async(std::launch::async,
                                 &ForestTrainer::train_batch,
                                 this,
                                 start_index,
                                 num_trees_batch,
                                 std::ref(data),
                                 options));
  }

  for (auto& future : futures) {
    std::vector<std::unique_ptr<Tree>> thread_trees = future.get();
    trees.insert(trees.end(),
                 std::make_move_iterator(thread_trees.begin()),
                 std::make_move_iterator(thread_trees.end()));
  }

  return trees;
}

std::vector<std::unique_ptr<Tree>> ForestTrainer::train_batch(
    size_t start,
    size_t num_trees,
    const Data& data,
    const ForestOptions& options) const {
  size_t ci_group_size = options.get_ci_group_size();

  // ----------------------------------------------
  // 将 ci_group_size 转化为 int 类型 
  // (在这之前我需要将其 nonlapping_block_size 和 其他函数的ci_group_size 转化为同一个变量，
  // 这里使用了 if_block 来判断是否是 block 抽样)
  
  bool if_block = options.get_if_block();
  int block_group_size = 0;
  if (if_block) {
    block_group_size = static_cast<int>(options.get_nonlapping_block_size());
  }else{
    block_group_size = static_cast<int>(ci_group_size);
  }
  // ----------------------------------------------

  std::mt19937_64 random_number_generator(options.get_random_seed() + start);
  // 创建一个均匀分布，用于生成随机数
  nonstd::uniform_int_distribution<uint> udist;
  // 创建保存树的向量
  std::vector<std::unique_ptr<Tree>> trees;

  // 预分配足够的空间，提高性能
  // trees.reserve(num_trees * ci_group_size);
  trees.reserve(num_trees);

  for (size_t i = 0; i < num_trees; i++) {
    uint tree_seed = udist(random_number_generator);

    // 定义一个随机采样器
    RandomSampler sampler(tree_seed, options.get_sampling_options());

    std::unique_ptr<Tree> tree = train_tree(data, sampler, options, block_group_size);
    trees.push_back(std::move(tree));
  }
  return trees;
}

// 训练单棵树
std::unique_ptr<Tree> ForestTrainer::train_tree(const Data& data,
                                                RandomSampler& sampler,
                                                const ForestOptions& options,
                                                int block_group_size) const {
  // cluster:动态数组，可自动管理其大小以适应存储的元素数量(无符号整型)，用于存储样本索引                                                
  std::vector<size_t> clusters;
  std::vector<std::vector<size_t>> blocks_clusters;

  sampler.sample_clusters(data.get_num_rows(), options.get_sample_fraction(), clusters, blocks_clusters, block_group_size);
  // 下面代码的作用：重新洗牌抽样，对clasters进行赋值修改
  /*  由于 clusters 是通过引用传递的，
  所以在 sample_clusters 方法内部所做的所有修改都会反映在外部传入的 clusters 向量中*/
  return tree_trainer.train(data, sampler, clusters, options.get_tree_options(), blocks_clusters);
}

// 训练置信区间组，进行多次抽样
std::vector<std::unique_ptr<Tree>> ForestTrainer::train_ci_group(const Data& data,
                                                                 RandomSampler& sampler,
                                                                 const ForestOptions& options,
                                                                 int block_group_size) const {
  std::vector<std::unique_ptr<Tree>> trees;

  std::vector<size_t> clusters;
  std::vector<std::vector<size_t>> blocks_clusters;

  // 第一次进行 默认为 0.5 的抽样
  sampler.sample_clusters(data.get_num_rows(), 0.5, clusters, blocks_clusters, block_group_size); // 调用 block 抽样

  double sample_fraction = options.get_sample_fraction();

  for (size_t i = 0; i < options.get_ci_group_size(); ++i) {
    std::vector<size_t> cluster_subsample;
    std::vector<std::vector<size_t>> blocks_clusters_subsample;
    // 二次抽样，按 sample_fraction*2 的比例进行抽样
    sampler.subsample_for_cigroup(clusters, blocks_clusters, sample_fraction * 2, cluster_subsample, blocks_clusters_subsample); 

    std::unique_ptr<Tree> tree = tree_trainer.train(data, sampler, cluster_subsample, options.get_tree_options(), blocks_clusters_subsample);
    trees.push_back(std::move(tree));
  }
  return trees;
}

} // namespace grf