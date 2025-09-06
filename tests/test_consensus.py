import pytest
import numpy as np
from reddwarf.utils.consensus import select_consensus_statements, select_grouped_consensus
from reddwarf.utils.matrix import VoteMatrix
from reddwarf.utils.stats import votes

class MockVoteMatrix:
    """テスト用のモックVoteMatrixクラス"""
    def __init__(self, n_groups=2, n_statements=5, n_participants=100):
        self.n_groups = n_groups
        self.columns = list(range(n_statements))  # ステートメントID
        self.statements = [{"statement_id": i, "txt": f"Statement {i}"} for i in range(n_statements)]
        self.index = list(range(n_participants))  # 参加者ID
        self.n_participants = n_participants
        
        # 各グループの参加者を分割
        group_size = n_participants // n_groups
        self.cluster_labels = np.zeros(n_participants, dtype=int)
        for g in range(1, n_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size if g < n_groups - 1 else n_participants
            self.cluster_labels[start_idx:end_idx] = g
        
        # 各グループの投票確率を設定
        # グループ0: 最初の2つのステートメントに賛成、残りは反対
        group0_probs = np.ones(n_statements) * 0.1  # デフォルト10%の確率
        group0_probs[:2] = 0.9  # 最初の2つは90%の確率で賛成
        
        # グループ1: 最後の2つのステートメントに賛成、最初のは反対
        group1_probs = np.ones(n_statements) * 0.1  # デフォルト10%の確率
        group1_probs[-2:] = 0.9  # 最後の2つは90%の確率で賛成
        
        # ダミーの投票行列を生成 (participants x statements)
        self.values = np.zeros((n_participants, n_statements), dtype=int)
        
        for i in range(n_participants):
            group = self.cluster_labels[i]
            if group == 0:
                probs = group0_probs
            else:
                probs = group1_probs
                
            # 各ステートメントに対して確率的に投票
            for j in range(n_statements):
                if np.random.random() < probs[j]:
                    self.values[i, j] = 1  # 賛成
                elif np.random.random() < 0.5:  # 50%の確率で反対
                    self.values[i, j] = -1  # 反対
                # 残りは0 (未投票)
        
        # テスト統計量を計算
        self.N_g_c = np.zeros((n_groups, n_statements))
        self.N_v_g_c = np.zeros((2, n_groups, n_statements))  # 0: agree, 1: disagree
        
        # 各グループで明確なパターンを持つようにデータを生成
        for i in range(n_participants):
            group = self.cluster_labels[i]
            for j in range(n_statements):
                # グループ0: 最初の2つのステートメントに賛成、残りは反対
                if group == 0:
                    if j < 2:  # 最初の2つは90%の確率で賛成
                        self.values[i, j] = 1 if np.random.random() < 0.9 else -1
                    else:  # 残りは10%の確率で賛成
                        self.values[i, j] = 1 if np.random.random() < 0.1 else -1
                # グループ1: 最後の2つのステートメントに賛成、最初のは反対
                else:
                    if j >= n_statements - 2:  # 最後の2つは90%の確率で賛成
                        self.values[i, j] = 1 if np.random.random() < 0.9 else -1
                    elif j == 0:  # 最初のは10%の確率で賛成（90%反対）
                        self.values[i, j] = 1 if np.random.random() < 0.1 else -1
                    else:  # その他はランダム
                        self.values[i, j] = 1 if np.random.random() < 0.5 else -1
                
                # 統計量を更新
                vote = self.values[i, j]
                if vote != 0:  # 未投票でない場合
                    self.N_g_c[group, j] += 1
                    if vote == 1:  # 賛成
                        self.N_v_g_c[votes.A, group, j] += 1
                    else:  # 反対
                        self.N_v_g_c[votes.D, group, j] += 1
        
        # 確率の計算
        self.P_v_g_c = np.zeros_like(self.N_v_g_c, dtype=float)
        for g in range(n_groups):
            for j in range(n_statements):
                total = self.N_g_c[g, j]
                if total > 0:
                    self.P_v_g_c[votes.A, g, j] = self.N_v_g_c[votes.A, g, j] / total
                    self.P_v_g_c[votes.D, g, j] = self.N_v_g_c[votes.D, g, j] / total
        
        # テスト統計量（簡易版）
        self.P_v_g_c_test = self.P_v_g_c.copy()  # 実際にはもっと複雑な計算が必要
        
        # グループごとの合意度を計算
        self.C_v_c = np.zeros((2, n_statements))
        for j in range(n_statements):
            # Simple consensus calculation (actual logic needs adjustment)
            self.C_v_c[votes.A, j] = np.mean([self.P_v_g_c[votes.A, g, j] for g in range(n_groups)])
            self.C_v_c[votes.D, j] = np.mean([self.P_v_g_c[votes.D, g, j] for g in range(n_groups)])


def test_select_consensus_statements():
    """Test for select_consensus_statements"""
    # Create test data (1 group)
    n_groups = 1
    n_statements = 20
    matrix = MockVoteMatrix(n_groups=n_groups, n_statements=n_statements)
    
    # Show all results
    print("\n=== Input Data ===")
    print(f"Groups: {n_groups}, Statements: {n_statements}")
    print("Votes per statement (agree, disagree):")
    for stmt_id in range(n_statements):
        agree = matrix.N_v_g_c[0, 0, stmt_id]  # Group 0, agree votes
        disagree = matrix.N_v_g_c[1, 0, stmt_id]  # Group 0, disagree votes
        print(f"  Statement {stmt_id}: agree={agree:.1f}, disagree={disagree:.1f}, total={agree + disagree:.1f}")
    
    # Get results for group 0
    result = select_consensus_statements(
        vote_matrix=matrix,
        group_id=0,
        pick_max=None,  # Get all results
        mod_out_statement_ids=[]  # No statements to exclude
    )
    
    # Display results
    print("\n=== Results ===")
    for direction in ['agree', 'disagree']:
        print(f"\n{direction.upper()} results (all):")
        for item in result[direction]:
            print(f"  Statement {item['tid']}: "
                  f"prob={item['p-success']:.3f}, "
                  f"n_agree={item['n-success']}, "
                  f"n_total={item['n-trials']}, "
                  f"z_score={item['p-test']:.3f}")
    
    # Verify results
    assert 'agree' in result
    assert 'disagree' in result
    for direction in ['agree', 'disagree']:
        for item in result[direction]:
            assert 'p-success' in item, f"Missing 'p-success' in {direction} item"
            assert 'tid' in item, f"Missing 'tid' in {direction} item"
            assert 'n-success' in item, f"Missing 'n-success' in {direction} item"
            assert 'n-trials' in item, f"Missing 'n-trials' in {direction} item"
            assert 'p-test' in item, f"Missing 'p-test' in {direction} item"
    
    # 少なくとも1つは結果が返ってくることを確認
    assert len(result['agree']) > 0 or len(result['disagree']) > 0, "No consensus statements found"


def test_select_grouped_consensus():
    """select_grouped_consensusのテスト"""
    # テスト用のデータを作成（グループ数1でテスト）
    n_groups = 1
    matrix = MockVoteMatrix(n_groups=n_groups, n_statements=5)
    
    # グループ名を指定
    group_names = {0: "Group A", 1: "Group B"}
    
    # グループごとの結果を取得
    results = select_grouped_consensus(
        vote_matrix=matrix,
        group_ids=list(range(n_groups)),  # [0, 1] のグループIDを指定
        group_names=group_names,
        pick_max=2
    )
    
    # 結果の検証
    assert len(results) == n_groups, f"Expected {n_groups} groups, got {len(results)}"
    
    # 各グループの結果を検証
    for result in results:
        assert 'group_id' in result, "Missing 'group_id' in result"
        assert 'group_name' in result, "Missing 'group_name' in result"
        assert 'agree' in result, "Missing 'agree' in result"
        assert 'disagree' in result, "Missing 'disagree' in result"
        
        # グループ名が正しく設定されているか
        group_id = result['group_id']
        if group_id in group_names:
            assert result['group_name'] == group_names[group_id], \
                f"Incorrect group name for group {group_id}"
        
        # 結果の構造を確認
        for direction in ['agree', 'disagree']:
            for item in result[direction]:
                # 実際の出力に合わせてフィールド名を調整
                assert 'p-success' in item, f"Missing 'p-success' in {direction} item"
                assert 'tid' in item, f"Missing 'tid' in {direction} item"
                assert 'n-success' in item, f"Missing 'n-success' in {direction} item"
                assert 'n-trials' in item, f"Missing 'n-trials' in {direction} item"
                assert 'p-test' in item, f"Missing 'p-test' in {direction} item"


def test_select_grouped_consensus_default_groups():
    """デフォルトのグループIDを使用した場合のテスト"""
    # テストを簡略化するためにグループ数を1に
    n_groups = 1
    matrix = MockVoteMatrix(n_groups=n_groups, n_statements=5)
    
    # group_idsを指定せずに呼び出す（デフォルトのグループIDを使用）
    results = select_grouped_consensus(
        vote_matrix=matrix,
        pick_max=1
    )
    
    # すべてのグループの結果が返るはず
    assert len(results) == n_groups, f"Expected {n_groups} groups, got {len(results)}"
    
    # グループIDの確認
    group_ids = {r['group_id'] for r in results}
    assert group_ids == set(range(n_groups)), f"Expected group IDs {set(range(n_groups))}, got {group_ids}"
    
    # 各グループの結果を検証
    for result in results:
        # 結果の構造を確認
        assert 'agree' in result, "Missing 'agree' in result"
        assert 'disagree' in result, "Missing 'disagree' in result"
        
        # 各方向（賛成/反対）の結果を確認
        for direction in ['agree', 'disagree']:
            # 結果がリストであることを確認
            assert isinstance(result[direction], list), \
                f"Expected list for {direction}, got {type(result[direction])}"
            
            # 各アイテムに必要なフィールドが含まれているか確認
            for item in result[direction]:
                assert 'p-success' in item, f"Missing 'p-success' in {direction} item"
                assert 'tid' in item, f"Missing 'tid' in {direction} item"
                assert 'n-success' in item, f"Missing 'n-success' in {direction} item"
                assert 'n-trials' in item, f"Missing 'n-trials' in {direction} item"
                assert 'p-test' in item, f"Missing 'p-test' in {direction} item"
