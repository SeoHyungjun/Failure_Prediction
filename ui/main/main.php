<!DOCTYPE html>
<html>
    <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.6/jquery.js"></script>
    <style type="text/css">
        #wrap {
            width : 1600px;
            height:1000px;
        }
        #header{
            width:900px;
            height:80px;
            margin-left:50px;
            margin-bottom:10px;
            align: center;
            text-align: center;
            vertical-align: middle;
        }
        #header-font{
            display: inline-block;
            line-height: 80px;
            font-size:2em;
        }
        #middle{
            width: 1300px;
            float: left;
        }
        #forced-graph{
            width : 740px;
            height : 500px;
            float : left;
            margin-left: 50px;
        }
        #usage-graph{
            width : 500px;
            height : 650px;
            float : left;
        }
        #log-table{
            width: 760px;
            height: 200px;
            float: left;
            margin-left: 50px;
            margin-top:  10px;
        }
    </style>
    <head>
        <meta charset="utf-8">
        <title>Data Visualization Term Project</title>
    </head>
    <body>
        <div id="wrap">
            <div id="header">
                <div id="header-font">
                    Data Visualization - Kim Youngwoo
                </div>
            </div>
            <div id="middle">
                <div id="forced-graph">
                    <iframe align="center" width="100%" height="100%" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="../forced_graph/forced_graph_cluster.html"></iframe>
                </div>
                <div id="usage-graph">
                    <iframe align="center" width="100%" height="100%" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="../usage_graph/usage_graph.html"></iframe>
                </div>
            </div>
            <script type="text/javascript">
                $(window).load(function () {
                    var $contents = $('#log-frame-id').contents();
                    $contents.scrollTop($contents.height());
                });
            </script>
            <div id="log-table">
                <iframe name="log-frame" id="log-frame-id" align="center" width="100%" height="100%" frameborder="0" scrolling="yes" marginheight="0" marginwidth="0" onload="scroll_to_end()" src="../log_track/log_track.php" ></iframe>
            </div>

        </div>
    </body>
</html>
