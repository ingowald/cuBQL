#!/usr/bin/python2.7

all_num_points = [ 1000, 10000, 100000, 1000000 ]
all_num_dims = [ 2, 3, 4 ]
#all_num_dims = [ 3 ]
box_generators = [
    "uniform",
    "clustered",
    # teapots:
    "mixture 100 uniform remap [ .1 .3 ]  [ .2 .4 ] clustered ",
    # outliers:
    "mixture 10 remap [ 3 4 ]  [ 5 5 ] clustered clustered",
    # whales:
    "mixture .001 clustered gaussian.scale 10 clustered",
    ]
point_generators = [
    "uniform",
    "clustered",
    # teapots:
    "mixture 100 uniform remap [ .1 .3 ]  [ .2 .4 ] clustered ",
    # outliers:
    "mixture 10 remap [ 3 4 ]  [ 5 5 ] clustered clustered",
    ]

#all_num_points = [ 10000 ]
#all_num_dims = [ 2, 3, 4 ]
#box_generators = [ box_generators[0] ]
#point_generators = [ point_generators[2] ]

def run(gen_id,gen_string,num_points,num_dims,data_type):
    name="n{npts}_gen{id}".format(npts=num_points,
                                        id=gen_id)
    if (num_dims == 2) :
        #
        plot_cmd="./cuBQL_plot{type} -dc {npts} -dg '{gen}' -o plot{type}_{pname}.svg".format(npts=num_points,dims=num_dims,id=gen_id,gen=gen_string,pname=name,type=data_type)
        print("echo '======================================================='")
        print("echo '"+plot_cmd+"'")
        print(plot_cmd)
        convert_cmd="convert plot{type}_{pname}.svg plot{type}_{pname}.png".format(pname=name,type=data_type)
        print("echo '"+convert_cmd+"'")
        print(convert_cmd)
    #
    # common stuffs
    #
    out_base="fcp{type}_nd{nd}_dc{dc}_dg{dg}".format(nd=num_dims,type=data_type,dg=gen_id,dc=num_points)
    base_cmd="./cuBQL_fcpAndKnn{type} -nd {dims} -dc {npts} -dg '{gen}' -qc 100000 -qg uniform".format(npts=num_points,dims=num_dims,id=gen_id,gen=gen_string,pname=name,type=data_type)
    #
    # default version, no elh, lt 1
    #
    fcp_cmd = base_cmd + " -lt 1"
    print("echo '-------------------------------------------------------'")
    print("echo '"+fcp_cmd+"'")
    print(fcp_cmd + " > out.txt")
    print("cat out.txt | grep STATS_DIGEST > "+out_base+"-lt1.stats")
    print("cat out.txt | grep STATS_DIGEST")
    #
    # default version, no elh, lt 4
    #
    fcp_cmd = base_cmd + " -lt 4"
    print("echo '--------------------------'")
    print("echo '"+fcp_cmd+"'")
    print(fcp_cmd + " > out.txt")
    print("cat out.txt | grep STATS_DIGEST > "+out_base+"-lt4.stats")
    print("cat out.txt | grep STATS_DIGEST")
    #
    # WITH elh
    #
    fcp_cmd = base_cmd + " -elh"
    print("echo '-------------------------'")
    print("echo '"+fcp_cmd+"'")
    print(fcp_cmd + " > out.txt")
    print("cat out.txt | grep STATS_DIGEST > "+out_base+"-elh.stats")
    print("cat out.txt | grep STATS_DIGEST")
    #
    # KD tree reference
    #
    if (num_dims <= 4) and (data_type == "Points") :
        print("echo '-------------------------'")
        print("echo '--- generating reference data ---'")
        fcp_cmd = base_cmd + " --dump-test-data"
        print(fcp_cmd + " > out.txt")
        kd_fcp_cmd = "../../cudaKDTree/bin/cukd_test_float{n}-fcp".format(n=num_dims)
        print("echo '"+kd_fcp_cmd+"'")
        print(kd_fcp_cmd + " > kd-out.txt")
        print("cat kd-out.txt | grep KDTREE_STATS > "+out_base+"-kd.stats")
        print("cat kd-out.txt | grep KDTREE_STATS")
    print("echo '##################################################################'")
    #
    # common stuffs
    #
    out_base="knn{type}_nd{nd}_dc{dc}_dg{dg}".format(nd=num_dims,type=data_type,dg=gen_id,dc=num_points)
    base_cmd="./cuBQL_fcpAndKnn{type} -knn-k 64-nd {dims} -dc {npts} -dg '{gen}' -qc 100000 -qg uniform".format(npts=num_points,dims=num_dims,id=gen_id,gen=gen_string,pname=name,type=data_type)
    #
    # default version, no elh, lt 1
    #
    fcp_cmd = base_cmd + " -lt 1"
    print("echo '-------------------------------------------------------'")
    print("echo '"+fcp_cmd+"'")
    print(fcp_cmd + " > out.txt")
    print("cat out.txt | grep STATS_DIGEST > "+out_base+"-lt1.stats")
    print("cat out.txt | grep STATS_DIGEST")
    #
    # default version, no elh, lt 4
    #
    fcp_cmd = base_cmd + " -lt 4"
    print("echo '--------------------------'")
    print("echo '"+fcp_cmd+"'")
    print(fcp_cmd + " > out.txt")
    print("cat out.txt | grep STATS_DIGEST > "+out_base+"-lt4.stats")
    print("cat out.txt | grep STATS_DIGEST")
    #
    # WITH elh
    #
    fcp_cmd = base_cmd + " -elh"
    print("echo '-------------------------'")
    print("echo '"+fcp_cmd+"'")
    print(fcp_cmd + " > out.txt")
    print("cat out.txt | grep STATS_DIGEST > "+out_base+"-elh.stats")
    print("cat out.txt | grep STATS_DIGEST")
    
    
def main():
    print("# scriptgenerated by makeTestScript.py")
    for num_points in all_num_points :
        for num_dims in all_num_dims :
            for gen_id in range(0, len(point_generators)) :
                gen_string = point_generators[gen_id]
                run(gen_id,gen_string,num_points,num_dims,'Points')
            for gen_id in range(0, len(box_generators)) :
                gen_string = box_generators[gen_id]
                run(gen_id,gen_string,num_points,num_dims,'Boxes')

main()
