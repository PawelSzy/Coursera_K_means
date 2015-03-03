"""
Template for Project 3
Student will implement four functions:

slow_closest_pairs(cluster_list)
fast_closest_pair(cluster_list) - implement fast_helper()
hierarchical_clustering(cluster_list, num_clusters)
kmeans_clustering(cluster_list, num_clusters, num_iterations)

where cluster_list is a list of clusters in the plane
"""

import math
import alg_cluster
import urllib2
import alg_cluster

   


def pair_distance(cluster_list, idx1, idx2):
    """
    Helper function to compute Euclidean distance between two clusters
    in cluster_list with indices idx1 and idx2
    
    Returns tuple (dist, idx1, idx2) with idx1 < idx2 where dist is distance between
    cluster_list[idx1] and cluster_list[idx2]
    """
    return (cluster_list[idx1].distance(cluster_list[idx2]), min(idx1, idx2), max(idx1, idx2))


def slow_closest_pairs(cluster_list):
    """
    Compute the set of closest pairs of cluster in list of clusters
    using O(n^2) all pairs algorithm
    
    Returns the set of all tuples of the form (dist, idx1, idx2) 
    where the cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.   
    
    """
#    print "\n----slow_closest_pairs---------\n"
#    print cluster_list
    
    closest_pairs = set([]) 
    
    pointers = [p for p in range(len(cluster_list))]
    
    for idx1 in pointers:
        closest_pair = (float('inf'), -1, -1)
        for idx2 in pointers:
            if idx1!=idx2:
             distance_cluster = pair_distance(cluster_list, idx1, idx2)
             if distance_cluster[0] < closest_pair[0]:
                    closest_pair = distance_cluster
        closest_pairs.update(([closest_pair]))
        
    return closest_pairs                             
                            

    return set([(0, 0, 0)])

def bf_closet_pair(cluster_list):
    """
    brute force find closet pair in the cluster list
    """
    closest_pairs = slow_closest_pairs(cluster_list)
    closet_pair = (float('inf'), -1, -1)
    for cluster in closest_pairs:
        if cluster[0] < closet_pair[0]:
            closet_pair = cluster
    return closet_pair       

def fast_closest_pair(cluster_list):
    """
    Compute a closest pair of clusters in cluster_list
    using O(n log(n)) divide and conquer algorithm
    
    Returns a tuple (distance, idx1, idx2) with idx1 < idx 2 where
    cluster_list[idx1] and cluster_list[idx2]
    have the smallest distance dist of any pair of clusters
    """
        
    def fast_helper(cluster_list, horiz_order, vert_order):
        """
        Divide and conquer method for computing distance between closest pair of points
        Running time is O(n * log(n))
        
        horiz_order and vert_order are lists of indices for clusters
        ordered horizontally and vertically
        
        Returns a tuple (distance, idx1, idx2) with idx1 < idx 2 where
        cluster_list[idx1] and cluster_list[idx2]
        have the smallest distance dist of any pair of clusters
    
        """
        
        # base case
        lenght_of_list = len(horiz_order)
        if lenght_of_list <= 3:
            if lenght_of_list==1:
                return cluster_list[horiz_order[0]]
  
            closets_pair = (float('inf'), -1, -1)
            
            for idx1 in horiz_order:
                for idx2 in horiz_order:
                    if idx1!=idx2:
                        distance_cluster = pair_distance(cluster_list, idx1, idx2)
                        if distance_cluster[0] < closets_pair[0]:
                            closets_pair = distance_cluster    

     
            return closets_pair
        
        
        
        # divide
        half = len(horiz_order)/2
        
#        print "\nhalf", half
        
        #print cluster_list[horiz_order[half-1]]
        
        horizontal_cord_half_minus_1 = cluster_list[horiz_order[half-1]].horiz_center()
        horizontal_cord_half = cluster_list[horiz_order[half]].horiz_center()


        mind_point = (horizontal_cord_half_minus_1+horizontal_cord_half)/2.0
#        print "half", half
#        print "mind_point", mind_point 
        
        horiz_list_e = horiz_order[0:half]
        horiz_list_r =horiz_order[half:]
        
#        print "cluster_list", cluster_list
#        print "horiz_list_e,horiz_list_r", horiz_list_e,horiz_list_r

        vert_list_e = []
        vert_list_r=[]
        
        for cluster in vert_order:
            if cluster in horiz_list_e:
                vert_list_e.append(cluster)
            else:
                vert_list_r.append(cluster)
                                
#        print "vert_list_e, vert_list_r", vert_list_e, vert_list_r
        
        closet_pair_vert_e =  fast_helper(cluster_list, horiz_list_e, vert_list_e) 
        closet_pair_vert_r = fast_helper(cluster_list, horiz_list_r, vert_list_r)
        
#        print "closet_pair_vert_e, closet_pair_vert_r", closet_pair_vert_e, closet_pair_vert_r
        
        if closet_pair_vert_e[0] < closet_pair_vert_r[0]:
            closet_pair = closet_pair_vert_e
        else: closet_pair = closet_pair_vert_r
            
#        print "closet_pair", closet_pair
        # conquer
        #Copy to S, in order, every V [i] for which |xV [i] & mid| < d
        s_list = []
        closet_distance = closet_pair[0]
#        print "\nvert_order", vert_order
#        print "mind_point", mind_point, "\n"
        for index_i in vert_order:
            v_center = cluster_list[index_i].horiz_center()
#            print "index, v_center", index_i, v_center
            if abs(v_center-mind_point) <closet_distance:
                s_list.append(index_i)
#        print "\ns_list", s_list
        
        #znajdz czy sa jakies mniejsze odleg³osci miedzy clusterami
        #w pasie pomiedzy punktem srodkowym
        lenght_s = len(s_list)
        for u_index in range(lenght_s-2):
            for v_index in range(u_index+1,min(u_index+3,lenght_s-1)):
                s_u_cluster = cluster_list[s_list[u_index]]
                s_v_cluster = cluster_list[s_list[v_index]]
                distance = s_u_cluster.distance(s_v_cluster)
                
                if distance < closet_pair[0]:
                    closet_pair= (distance, u_index, v_index)


        
        return closet_pair
            
    # compute list of indices for the clusters ordered in the horizontal direction
    hcoord_and_index = [(cluster_list[idx].horiz_center(), idx) 
                        for idx in range(len(cluster_list))]    
    hcoord_and_index.sort()
    horiz_order = [hcoord_and_index[idx][1] for idx in range(len(hcoord_and_index))]
     
    # compute list of indices for the clusters ordered in vertical direction
    vcoord_and_index = [(cluster_list[idx].vert_center(), idx) 
                        for idx in range(len(cluster_list))]    
    vcoord_and_index.sort()
    vert_order = [vcoord_and_index[idx][1] for idx in range(len(vcoord_and_index))]

    # compute answer recursively
    answer = fast_helper(cluster_list, horiz_order, vert_order) 
    return (answer[0], min(answer[1:]), max(answer[1:]))

def center_of_cluster_list(cluster_list):
    """
    take cluster list 
    return center of cluster (x,y)
    """
    points = []
    for cluster in cluster_list:
        point = cluster.horiz_center(), cluster.vert_center()
        points.append(point)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points)*1.0, sum(y) / len(points))*1.0  
    return centroid

def hierarchical_clustering(cluster_list, num_clusters):
    """
    Compute a hierarchical clustering of a set of clusters
    Note: the function mutates cluster_list
    
    Input: List of clusters, number of clusters
    Output: List of clusters whose length is num_clusters
    """
    cluster_list = list(cluster_list)
    
#    print "ddd"
#    print cluster_list
#    print "len", len(cluster_list), "\n"
    while len(cluster_list) > num_clusters:

        closet_pair = fast_closest_pair(cluster_list)
        closet_pair_bf = bf_closet_pair(cluster_list)
#        print "fast_closest_pair", closet_pair
#        print "bf_closet_pair", closet_pair_bf
        
        cluster1= cluster_list[closet_pair[1]]
        cluster2= cluster_list[closet_pair[2]]
        cluster1.merge_clusters(cluster2)
        
        del cluster_list[closet_pair[2]]

        
        
    return cluster_list

def distance_two_points(horiz_center1, vert_center1,horiz_center2, vert_center2):
        vert_dist = vert_center1 - vert_center2
        horiz_dist = horiz_center1 - horiz_center2
        return math.sqrt(vert_dist ** 2 + horiz_dist ** 2)
    
def kmeans_clustering(cluster_list, num_clusters, num_iterations):
    """
    Compute the k-means clustering of a set of clusters
    
    Input: List of clusters, number of clusters, number of iterations
    Output: List of clusters whose length is num_clusters
    """
    
    def cluster_min_population(cluster_list):
        """
        Find cluster with minimal population
        """
        min_pop_cluster = alg_cluster.Cluster(0,0,0,float("inf"),0)
        for cluster in cluster_list:
            if min_pop_cluster.total_population() > cluster.total_population():
                min_pop_cluster = cluster
        return  min_pop_cluster    
    
    
    def find_cluster_with_min_distance(cluster, centers):
        """
        take cluster  
        return (index_of closet_center, distance, center)
        """

        return_index = (0, float("inf"), (0,0))
        for index in range(len(centers)):
            hor_cluster, vert_cluster = cluster.horiz_center(), cluster.vert_center()
            distance = distance_two_points(hor_cluster, vert_cluster, centers[index][0],centers[index][1])
            if distance < return_index[1]:
                return_index = (index, distance, centers[index])
        return return_index        
    
    # initialize k-means clusters to be initial clusters with largest populations
    largest_pop = [0]*num_clusters
    initial_clusters = set([])
    for cluster in cluster_list:
        if len(initial_clusters) < num_clusters:
            initial_clusters.add(cluster)
            continue
            
        min_cluster = cluster_min_population(initial_clusters)
        if cluster.total_population() > min_cluster.total_population():
            initial_clusters.add(cluster)
            initial_clusters.remove(min_cluster)
            
            
    #initialize centers of clusters - cluster with larges population
    centers= []
    for cluster in initial_clusters:
        centers.append((cluster.horiz_center(), cluster.vert_center()))
    
    #k-means clustering - main algorytm
    len_cluster_list = len(cluster_list)
    for iteration in range(num_iterations):
        new_sets_of_clusters = [set([]) for x_dummy in range(num_clusters)]
        #find center point with minimal distance to cluster 
        for cluster in cluster_list:
            pass
            #index_of_min_dist_cluster  (index_of closet_center, distance, center)
            min_dist_center =find_cluster_with_min_distance(cluster, centers)     
            index_of_min_dist_center =min_dist_center[0]
#            print "index_of_min_dist_center", index_of_min_dist_center
#            print "cluster", cluster
            new_sets_of_clusters[index_of_min_dist_center].add(cluster)
        for index_centers in range(len(centers)):
             clust_list = new_sets_of_clusters[index_centers]
             centers[index_centers] = center_of_cluster_list(clust_list)
#    print "new_sets_of_clusters", new_sets_of_clusters
#    print len(new_sets_of_clusters)
#    print "centers", centers
    return new_sets_of_clusters


def test_run():
    """
    Testing function 
    """
    #Cluster(FIPS_codes, horiz_center, vert_center, total_population, average_risk)
    
    clusters_text1 = ["01069, 740.09731366, 463.241137095, 88787, 4.0E-05",
    "01067, 741.064829551, 454.67645286, 16310, 3.4E-05",
    "01061, 730.413538241, 465.838757711, 25764, 3.8E-05",
    "01045, 733.967850833, 457.849623249, 49129, 3.9E-05",
    "01031, 726.661721748, 459.039231303, 43615, 3.8E-05",
    "01039, 717.962710784, 463.38408665, 37631, 3.7E-05",
    "01097, 684.031627259, 477.419999644, 399843, 4.2E-05",
    "01025, 690.135425828, 457.231431451, 27867, 3.6E-05"]
    
    
    
    clusters_text2 = ["0, 0, 100, 5, 5",
                      "1, 1, 2, 10, 5", 
                      "2, 1, 30, 20, 5", 
                      "3, 1, 31, 5, 5", 
                      "4, 4, 34, 5, 5", 
                      "5, 4, 36, 5, 5", 
                      "6, 14, 1, 5, 5", 
                      "7, 50, 30, 5, 5", 
                      "8, 50, 33, 5, 5", 
                      "9, 53, 29, 30, 5", 
                      "10, 53, 35, 5, 5"]
    
    clusters_text3 = ["1, 1, 2, 5, 5", 
 
                      "7, 50, 30, 5, 5", 
                      "8, 50, 33, 5, 5", 
                      "9, 53, 29, 5, 5", 
                      "10, 53, 35, 5, 5"]
  

    
    clusters_text = clusters_text2

    clusters_list = []
    
    for cluster in clusters_text:
        cluster = cluster.replace(",", "")
        print "cluster", cluster
        words = cluster.split()
        print words
        #num_words[] = [float(x_dummy) for x_dummy in words[1:]]
        clusters_list.append(alg_cluster.Cluster(set([words[0]]), float(words[1]),  float(words[2]),  int(words[3]),  float(words[4])))

    print  "clusters_list", clusters_list, "\n"   
        
    print "\nslow_closest_pairs", slow_closest_pairs(clusters_list), "\n"   
    
    print "bf_closet_pair", bf_closet_pair(clusters_list)    
    
    print "\nfast_closest_pair", fast_closest_pair(clusters_list)
    
    num_clusters = 4
    print "\nhierarchical_clustering"

#    print hierarchical_clustering(clusters_list, num_clusters)
    
    
    num_iterations = 5
    print "\nK-means clustering"
    
    
    print kmeans_clustering(clusters_list, num_clusters, num_iterations)
test_run()    