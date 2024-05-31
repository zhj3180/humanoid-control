/******************************************************************************
Copyright (c) 2021, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include <iostream>
#include <string>

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "humanoid_interface/HumanoidInterface.h"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_core/misc/Display.h>
#include <ocs2_core/misc/LoadStdVectorOfPair.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>

#include "humanoid_interface/HumanoidPreComputation.h"
#include "humanoid_interface/constraint/FrictionConeConstraint.h"
#include "humanoid_interface/constraint/NormalVelocityConstraintCppAd.h"
#include "humanoid_interface/constraint/ZeroForceConstraint.h"
#include "humanoid_interface/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_interface/constraint/FootRollConstraint.h"
#include "humanoid_interface/constraint/LeggedSelfCollisionConstraint.h"
#include "humanoid_interface/cost/HumanoidQuadraticTrackingCost.h"
#include "humanoid_interface/dynamics/HumanoidDynamicsAD.h"

// Boost
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#define threedof 1

namespace ocs2 {
namespace humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
HumanoidInterface::HumanoidInterface(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile,
                                           bool useHardFrictionConeConstraint)
    : useHardFrictionConeConstraint_(useHardFrictionConeConstraint) {
  // check that task file exists
  boost::filesystem::path taskFilePath(taskFile);
  if (boost::filesystem::exists(taskFilePath)) {
    std::cerr << "[HumanoidInterface] Loading task file: " << taskFilePath << std::endl;
  } else {
    throw std::invalid_argument("[HumanoidInterface] Task file not found: " + taskFilePath.string());
  }
  // check that urdf file exists
  boost::filesystem::path urdfFilePath(urdfFile);
  if (boost::filesystem::exists(urdfFilePath)) {
    std::cerr << "[HumanoidInterface] Loading Pinocchio model from: " << urdfFilePath << std::endl;
  } else {
    throw std::invalid_argument("[HumanoidInterface] URDF file not found: " + urdfFilePath.string());
  }
  // check that targetCommand file exists
  boost::filesystem::path referenceFilePath(referenceFile);
  if (boost::filesystem::exists(referenceFilePath)) {
    std::cerr << "[HumanoidInterface] Loading target command settings from: " << referenceFilePath << std::endl;
  } else {
    throw std::invalid_argument("[HumanoidInterface] targetCommand file not found: " + referenceFilePath.string());
  }

  bool verbose;
  loadData::loadCppDataType(taskFile, "humanoid_interface.verbose", verbose);

  // load setting from loading file
  modelSettings_ = loadModelSettings(taskFile, "model_settings", verbose);
  mpcSettings_ = mpc::loadSettings(taskFile, "mpc", verbose);
  ddpSettings_ = ddp::loadSettings(taskFile, "ddp", verbose);
  sqpSettings_ = sqp::loadSettings(taskFile, "sqp", verbose);
  ipmSettings_ = ipm::loadSettings(taskFile, "ipm", verbose);
  rolloutSettings_ = rollout::loadSettings(taskFile, "rollout", verbose);

  // OptimalConrolProblem
  setupOptimalControlProblem(taskFile, urdfFile, referenceFile, verbose);

  // initial state
  initialState_.setZero(centroidalModelInfo_.stateDim);
  loadData::loadEigenMatrix(taskFile, "initialState", initialState_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void HumanoidInterface::setupOptimalControlProblem(const std::string& taskFile, const std::string& urdfFile,
                                                     const std::string& referenceFile, bool verbose) {
  // PinocchioInterface
  pinocchioInterfacePtr_.reset(new PinocchioInterface(centroidal_model::createPinocchioInterface(urdfFile, modelSettings_.jointNames)));

  //*****************************************************************************************************************************************
  // CentroidalModelInfo
  // 创建中心模型info，info结构在cnetroidModelInfo.h中,获取模型的info.centroidalInertiaNominal，info.comToBasePositionNominal
  //中心惯性矩阵，指的是机器人惯性张量，3x3，J，转动惯量（角动量）=惯性张亮*角速度
  //质心到基座位置变换向量 3*1，xyz
  centroidalModelInfo_ = centroidal_model::createCentroidalModelInfo(
      *pinocchioInterfacePtr_, centroidal_model::loadCentroidalType(taskFile),
      centroidal_model::loadDefaultJointState(pinocchioInterfacePtr_->getModel().nq - 6, referenceFile), modelSettings_.contactNames3DoF,
      modelSettings_.contactNames6DoF); //type:Single Rigid Body Dynamics
  // Swing trajectory planner
  //摆腿轨迹设定，只有z方向的位置与速度，也就是高度，三次曲线组合，通过设定中间高度组合
  auto swingTrajectoryPlanner =
      std::make_unique<SwingTrajectoryPlanner>(loadSwingTrajectorySettings(taskFile, "swing_trajectory_config", verbose), 4);

  //*****************************************************************************************************************************************
  // Mode schedule manager
  //步态规划  /humanoid_interface/gait/motionPhaseDefinition.h ，在里面将两只脚的四个接触点规划，顺边
  referenceManagerPtr_ =
      std::make_shared<SwitchedModelReferenceManager>(loadGaitSchedule(referenceFile, verbose), std::move(swingTrajectoryPlanner));

  // Optimal control problem
  //初始化优化问题
  problemPtr_.reset(new OptimalControlProblem);

  //*****************************************************************************************************************************************
  // Dynamics
  bool useAnalyticalGradientsDynamics = false;    //梯度优化，相较于数值逼近，对导数求解更精确，没有使用
  loadData::loadCppDataType(taskFile, "humanoid_interface.useAnalyticalGradientsDynamics", useAnalyticalGradientsDynamics);
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  if (useAnalyticalGradientsDynamics) {
    throw std::runtime_error("[HumanoidInterface::setupOptimalControlProblem] The analytical dynamics class is not yet implemented!");
  } else {
    const std::string modelName = "dynamics";
    //加载Pinocchio模型结构，中心模型相关参数
    //动力学方程指针dynamicsPtr的构造函数使用的是pinocchioCentroidalDynamicsAd_，在ocs2_centroidal_model/PinocchioCentroidalDynamicsAD.h中
    //使用CppAD自动生成动力学计算函数，x_dot=f(x,u);  函数放在task.info指定的文件夹中 /tmp/ocs2 通过中心模型参数，直接生成函数文件
    //函数内部计算无法明确获取，只能获取结果
    //该指针主要功能是调用，数据流图计算函数，得到结果
    dynamicsPtr.reset(new HumanoidDynamicsAD(*pinocchioInterfacePtr_, centroidalModelInfo_, modelName, modelSettings_));
  }

  problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

  //*****************************************************************************************************************************************
  // Cost terms
  //读取task.info设定参数，得到Q矩阵 24*24， 计算雅可比矩阵， 得到R矩阵 24*24，l_force,l_torque,r_force,r_torque,v_(1~12)
  problemPtr_->costPtr->add("baseTrackingCost", getBaseTrackingCost(taskFile, centroidalModelInfo_, false));

  //*****************************************************************************************************************************************
  // Constraint terms
  // friction cone settings
  scalar_t frictionCoefficient = 0.7;
  RelaxedBarrierPenalty::Config barrierPenaltyConfig;
  std::tie(frictionCoefficient, barrierPenaltyConfig) = loadFrictionConeSettings(taskFile, verbose);

  bool useAnalyticalGradientsConstraints = false;
  loadData::loadCppDataType(taskFile, "humanoid_interface.useAnalyticalGradientsConstraints", useAnalyticalGradientsConstraints);
#if threedof
  for (size_t i = 0; i < centroidalModelInfo_.numThreeDofContacts; i++) {
    const std::string& footName = modelSettings_.contactNames3DoF[i];
#else
  for (size_t i = 0; i < centroidalModelInfo_.numSixDofContacts; i++) {
    const std::string& footName = modelSettings_.contactNames6DoF[i];
#endif
  //创建末端执行器运动学指针，在ocs2_robot_tools里 EndEfffectorKinematics.h
  //计算末端执行器速度，位置
    std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr;
    if (useAnalyticalGradientsConstraints) {
      throw std::runtime_error(
          "[HumanoidInterface::setupOptimalControlProblem] The analytical end-effector linear constraint is not implemented!");
    } else {
      const auto infoCppAd = centroidalModelInfo_.toCppAd();
      const CentroidalModelPinocchioMappingCppAd pinocchioMappingCppAd(infoCppAd);
      auto velocityUpdateCallback = [&infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pinocchioInterfaceAd) {
        const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
        updateCentroidalDynamics(pinocchioInterfaceAd, infoCppAd, q);


  // std::ofstream outFile_state("state.txt", std::ios::trunc);
  // if (!outFile_state.is_open()) {
  //   std::cerr << "无法打开文件0" << std::endl;
  // }
  // else{
  //   outFile_state<<time<<std::endl;
  //   for(int i=0;i<state.size();i++)
  //     outFile_state<<state[i]<<" ";
  //   outFile_state<<std::endl;
  // }
  // outFile_state.close();

  //   std::ofstream outFile_q("q.txt", std::ios::trunc);
  // if (!outFile_q.is_open()) {
  //   std::cerr << "无法打开文件0" << std::endl;
  // }
  // else{
  //   outFile_q<<time<<std::endl;
  //   for(int i=0;i<q.size();i++)
  //     outFile_q<<q[i]<<" ";
  //   outFile_q<<std::endl;
  // }
  // outFile_q.close();

      };


    
      eeKinematicsPtr.reset(new PinocchioEndEffectorKinematicsCppAd(*pinocchioInterfacePtr_, pinocchioMappingCppAd, {footName},
                                                                    centroidalModelInfo_.stateDim, centroidalModelInfo_.inputDim,
                                                                    velocityUpdateCallback, footName, modelSettings_.modelFolderCppAd,
                                                                    modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd));
    }

    if (useHardFrictionConeConstraint_) {
      problemPtr_->inequalityConstraintPtr->add(footName + "_frictionCone", getFrictionConeConstraint(i, frictionCoefficient));
    } else {
      problemPtr_->softConstraintPtr->add(footName + "_frictionCone",
                                          getFrictionConeSoftConstraint(i, frictionCoefficient, barrierPenaltyConfig));
    }
    problemPtr_->equalityConstraintPtr->add(footName + "_zeroForce", getZeroForceConstraint(i));
    problemPtr_->equalityConstraintPtr->add(footName + "_zeroVelocity",
                                            getZeroVelocityConstraint(*eeKinematicsPtr, i, useAnalyticalGradientsConstraints));
    problemPtr_->equalityConstraintPtr->add(footName + "_normalVelocity",
                                            getNormalVelocityConstraint(*eeKinematicsPtr, i, useAnalyticalGradientsConstraints));
    // 由于一只脚上有两个虚拟接触点，只需要给其中一个接触点添加滚转角约束
      if (i < 2){
          problemPtr_->equalityConstraintPtr->add(footName + "_footRoll", getFootRollConstraint(i));
      }
    }
            // Self-collision avoidance constraint
    problemPtr_->stateSoftConstraintPtr->add("selfCollision",
                                                     getSelfCollisionConstraint(*pinocchioInterfacePtr_, taskFile, "selfCollision", verbose));

  // Pre-computation
  problemPtr_->preComputationPtr.reset(new HumanoidPreComputation(*pinocchioInterfacePtr_, centroidalModelInfo_,
                                                                     *referenceManagerPtr_->getSwingTrajectoryPlanner(), modelSettings_));

                                                                       std::cout<<std::endl;

  // Rollout
  rolloutPtr_.reset(new TimeTriggeredRollout(*problemPtr_->dynamicsPtr, rolloutSettings_));

  // Initialization
  constexpr bool extendNormalizedMomentum = true;
  initializerPtr_.reset(new HumanoidInitializer(centroidalModelInfo_, *referenceManagerPtr_, extendNormalizedMomentum));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::shared_ptr<GaitSchedule> HumanoidInterface::loadGaitSchedule(const std::string& file, bool verbose) const {
  const auto initModeSchedule = loadModeSchedule(file, "initialModeSchedule", false);
  const auto defaultModeSequenceTemplate = loadModeSequenceTemplate(file, "defaultModeSequenceTemplate", false);

  const auto defaultGait = [&] {
    Gait gait{};
    gait.duration = defaultModeSequenceTemplate.switchingTimes.back();
    // Events: from time -> phase
    std::for_each(defaultModeSequenceTemplate.switchingTimes.begin() + 1, defaultModeSequenceTemplate.switchingTimes.end() - 1,
                  [&](double eventTime) { gait.eventPhases.push_back(eventTime / gait.duration); });
    // Modes:
    gait.modeSequence = defaultModeSequenceTemplate.modeSequence;
    return gait;
  }();

  // display
  if (verbose) {
    std::cerr << "\n#### Modes Schedule: ";
    std::cerr << "\n#### =============================================================================\n";
    std::cerr << "Initial Modes Schedule: \n" << initModeSchedule;
    std::cerr << "Default Modes Sequence Template: \n" << defaultModeSequenceTemplate;
    std::cerr << "#### =============================================================================\n";
  }

  return std::make_shared<GaitSchedule>(initModeSchedule, defaultModeSequenceTemplate, modelSettings_.phaseTransitionStanceTime);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t HumanoidInterface::initializeInputCostWeight(const std::string& taskFile, const CentroidalModelInfo& info) {
  matrix_t R = matrix_t::Zero(info.inputDim, info.inputDim);
#if threedof
    const size_t totalContactDim = 3 * info.numThreeDofContacts;

    vector_t initialState(centroidalModelInfo_.stateDim);
    loadData::loadEigenMatrix(taskFile, "initialState", initialState);

    const auto& model = pinocchioInterfacePtr_->getModel();
    auto& data = pinocchioInterfacePtr_->getData();
    const auto q = centroidal_model::getGeneralizedCoordinates(initialState, centroidalModelInfo_);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    matrix_t baseToFeetJacobians(totalContactDim, info.actuatedDofNum);
    for (size_t i = 0; i < info.numThreeDofContacts; i++) {
        matrix_t jacobianWorldToContactPointInWorldFrame = matrix_t::Zero(6, info.generalizedCoordinatesNum);
        pinocchio::getFrameJacobian(model, data, model.getBodyId(modelSettings_.contactNames3DoF[i]), pinocchio::LOCAL_WORLD_ALIGNED,
                                    jacobianWorldToContactPointInWorldFrame);

        baseToFeetJacobians.block(3 * i, 0, 3, info.actuatedDofNum) =
                jacobianWorldToContactPointInWorldFrame.block(0, 6, 3, info.actuatedDofNum);
    }

    std::cout<<std::endl;
    std::cout<<"info.actuatedDofNum:::::"<<info.actuatedDofNum<<std::endl;
    std::cout<<std::endl;

    std::cout<<std::endl;
    std::cout<<"totalContactDim:::::"<<totalContactDim<<std::endl;
    std::cout<<std::endl;
    
    matrix_t R_taskspace(totalContactDim + totalContactDim, totalContactDim + totalContactDim);
    loadData::loadEigenMatrix(taskFile, "R", R_taskspace);

    // Contact Forces
    R.topLeftCorner(totalContactDim, totalContactDim) = R_taskspace.topLeftCorner(totalContactDim, totalContactDim);
    // Joint velocities
    R.bottomRightCorner(info.actuatedDofNum, info.actuatedDofNum) =
            baseToFeetJacobians.transpose() * R_taskspace.bottomRightCorner(totalContactDim, totalContactDim) * baseToFeetJacobians;
  std::ofstream outFile_jacobian("jacobian.txt", std::ios::trunc);
  if (!outFile_jacobian.is_open()) {
    std::cerr << "无法打开文件0" << std::endl;
  }
  else{
    outFile_jacobian<<time<<std::endl;
    outFile_jacobian<<R<<std::endl;
    outFile_jacobian<<std::endl;
  }
  outFile_jacobian.close();
#else
    const size_t totalContactDim = 6 * info.numSixDofContacts;

    vector_t initialState(centroidalModelInfo_.stateDim);
    loadData::loadEigenMatrix(taskFile, "initialState", initialState);

    const auto& model = pinocchioInterfacePtr_->getModel();
    auto& data = pinocchioInterfacePtr_->getData();
    const auto q = centroidal_model::getGeneralizedCoordinates(initialState, centroidalModelInfo_);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    matrix_t baseToFeetJacobians(totalContactDim, info.actuatedDofNum);
    for (size_t i = 0; i < info.numSixDofContacts; i++) {
        matrix_t jacobianWorldToContactPointInWorldFrame = matrix_t::Zero(6, info.generalizedCoordinatesNum);
        pinocchio::getFrameJacobian(model, data, model.getBodyId(modelSettings_.contactNames6DoF[i]), pinocchio::LOCAL_WORLD_ALIGNED,
                                    jacobianWorldToContactPointInWorldFrame);

        baseToFeetJacobians.block(6 * i, 0, 6, info.actuatedDofNum) =
                jacobianWorldToContactPointInWorldFrame.block(0, 6, 6, info.actuatedDofNum);
    }
    // std::cout<<std::endl;
    // std::cout<<"info.actuatedDofNum:::::"<<info.actuatedDofNum<<std::endl;
    // std::cout<<std::endl;

    // std::cout<<std::endl;
    // std::cout<<"totalContactDim:::::"<<totalContactDim<<std::endl;
    // std::cout<<std::endl;
    
    matrix_t R_taskspace(totalContactDim + totalContactDim, totalContactDim + totalContactDim);
    loadData::loadEigenMatrix(taskFile, "R", R_taskspace);

    // Contact Forces
    R.topLeftCorner(totalContactDim, totalContactDim) = R_taskspace.topLeftCorner(totalContactDim, totalContactDim);
    // Joint velocities
    R.bottomRightCorner(info.actuatedDofNum, info.actuatedDofNum) =
            baseToFeetJacobians.transpose() * R_taskspace.bottomRightCorner(totalContactDim, totalContactDim) * baseToFeetJacobians;
  std::ofstream outFile_jacobian_6("jacobian_6.txt", std::ios::trunc);
  if (!outFile_jacobian_6.is_open()) {
    std::cerr << "无法打开文件0" << std::endl;
  }
  else{
    outFile_jacobian_6<<time<<std::endl;
    outFile_jacobian_6<<R<<std::endl;
    outFile_jacobian_6<<std::endl;
  }
  outFile_jacobian_6.close();
#endif

    return R;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputCost> HumanoidInterface::getBaseTrackingCost(const std::string& taskFile, const CentroidalModelInfo& info,
                                                                          bool verbose) {
  matrix_t Q(info.stateDim, info.stateDim);
  loadData::loadEigenMatrix(taskFile, "Q", Q);
  matrix_t R = initializeInputCostWeight(taskFile, info);

  if (verbose) {
    std::cerr << "\n #### Base Tracking Cost Coefficients: ";
    std::cerr << "\n #### =============================================================================\n";
    std::cerr << "Q:\n" << Q << "\n";
    std::cerr << "R:\n" << R << "\n";
    std::cerr << " #### =============================================================================\n";
  }

  return std::make_unique<HumanoidStateInputQuadraticCost>(std::move(Q), std::move(R), info, *referenceManagerPtr_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::pair<scalar_t, RelaxedBarrierPenalty::Config> HumanoidInterface::loadFrictionConeSettings(const std::string& taskFile,
                                                                                                  bool verbose) const {
  boost::property_tree::ptree pt;
  boost::property_tree::read_info(taskFile, pt);
  const std::string prefix = "frictionConeSoftConstraint.";

  scalar_t frictionCoefficient = 1.0;
  RelaxedBarrierPenalty::Config barrierPenaltyConfig;
  if (verbose) {
    std::cerr << "\n #### Friction Cone Settings: ";
    std::cerr << "\n #### =============================================================================\n";
  }
  loadData::loadPtreeValue(pt, frictionCoefficient, prefix + "frictionCoefficient", verbose);
  loadData::loadPtreeValue(pt, barrierPenaltyConfig.mu, prefix + "mu", verbose);
  loadData::loadPtreeValue(pt, barrierPenaltyConfig.delta, prefix + "delta", verbose);
  if (verbose) {
    std::cerr << " #### =============================================================================\n";
  }

  return {frictionCoefficient, std::move(barrierPenaltyConfig)};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> HumanoidInterface::getFrictionConeConstraint(size_t contactPointIndex,
                                                                                      scalar_t frictionCoefficient) {
  FrictionConeConstraint::Config frictionConeConConfig(frictionCoefficient);
  return std::make_unique<FrictionConeConstraint>(*referenceManagerPtr_, std::move(frictionConeConConfig), contactPointIndex,
                                                  centroidalModelInfo_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputCost> HumanoidInterface::getFrictionConeSoftConstraint(
    size_t contactPointIndex, scalar_t frictionCoefficient, const RelaxedBarrierPenalty::Config& barrierPenaltyConfig) {
  return std::make_unique<StateInputSoftConstraint>(getFrictionConeConstraint(contactPointIndex, frictionCoefficient),
                                                    std::make_unique<RelaxedBarrierPenalty>(barrierPenaltyConfig));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> HumanoidInterface::getZeroForceConstraint(size_t contactPointIndex) {
  return std::make_unique<ZeroForceConstraint>(*referenceManagerPtr_, contactPointIndex, centroidalModelInfo_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> HumanoidInterface::getZeroVelocityConstraint(const EndEffectorKinematics<scalar_t>& eeKinematics,
                                                                                      size_t contactPointIndex,
                                                                                      bool useAnalyticalGradients) {
  auto eeZeroVelConConfig = [](scalar_t positionErrorGain) {
    EndEffectorLinearConstraint::Config config;
    config.b.setZero(3);
    config.Av.setIdentity(3, 3);
    if (!numerics::almost_eq(positionErrorGain, 0.0)) {
      config.Ax.setZero(3, 3);
      config.Ax(2, 2) = positionErrorGain;
    }
    return config;
  };

  if (useAnalyticalGradients) {
    throw std::runtime_error(
        "[HumanoidInterface::getZeroVelocityConstraint] The analytical end-effector zero velocity constraint is not implemented!");
  } else {
    return std::make_unique<ZeroVelocityConstraintCppAd>(*referenceManagerPtr_, eeKinematics, contactPointIndex,
                                                         eeZeroVelConConfig(modelSettings_.positionErrorGain));
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> HumanoidInterface::getNormalVelocityConstraint(const EndEffectorKinematics<scalar_t>& eeKinematics,
                                                                                        size_t contactPointIndex,
                                                                                        bool useAnalyticalGradients) {
  if (useAnalyticalGradients) {
    throw std::runtime_error(
        "[HumanoidInterface::getNormalVelocityConstraint] The analytical end-effector normal velocity constraint is not implemented!");
  } else {
    return std::make_unique<NormalVelocityConstraintCppAd>(*referenceManagerPtr_, eeKinematics, contactPointIndex);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> HumanoidInterface::getFootRollConstraint(size_t contactPointIndex){
    return std::make_unique<FootRollConstraint>(*referenceManagerPtr_, contactPointIndex, centroidalModelInfo_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateCost> HumanoidInterface::getSelfCollisionConstraint(const PinocchioInterface& pinocchioInterface,
                                                                       const std::string& taskFile, const std::string& prefix,
                                                                       bool verbose) {
    std::vector<std::pair<size_t, size_t>> collisionObjectPairs;
    std::vector<std::pair<std::string, std::string>> collisionLinkPairs;
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;
    scalar_t minimumDistance = 0.0;

    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);
    if (verbose) {
        std::cerr << "\n #### SelfCollision Settings: ";
        std::cerr << "\n #### =============================================================================\n";
    }
    loadData::loadPtreeValue(pt, mu, prefix + ".mu", verbose);
    loadData::loadPtreeValue(pt, delta, prefix + ".delta", verbose);
    loadData::loadPtreeValue(pt, minimumDistance, prefix + ".minimumDistance", verbose);
    loadData::loadStdVectorOfPair(taskFile, prefix + ".collisionObjectPairs", collisionObjectPairs, verbose);
    loadData::loadStdVectorOfPair(taskFile, prefix + ".collisionLinkPairs", collisionLinkPairs, verbose);

    geometryInterfacePtr_ = std::make_unique<PinocchioGeometryInterface>(pinocchioInterface, collisionLinkPairs, collisionObjectPairs);
    if (verbose) {
        std::cerr << " #### =============================================================================\n";
        const size_t numCollisionPairs = geometryInterfacePtr_->getNumCollisionPairs();
        std::cerr << "SelfCollision: Testing for " << numCollisionPairs << " collision pairs\n";
    }

    std::unique_ptr<StateConstraint> constraint = std::make_unique<LeggedSelfCollisionConstraint>(
            CentroidalModelPinocchioMapping(centroidalModelInfo_), *geometryInterfacePtr_, minimumDistance);

    auto penalty = std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config{mu, delta});

    return std::make_unique<StateSoftConstraint>(std::move(constraint), std::move(penalty));
}

}  // namespace humanoid
}  // namespace ocs2
